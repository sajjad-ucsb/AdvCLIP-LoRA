import torch
import torch.nn.functional as F

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, \
    load_lora
from loralib import layers as lora_layers


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image


def evaluate_lora(args, clip_model, loader, dataset, attack="True"):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    # with torch.no_grad():
    for i, (images, target) in enumerate(loader):
        images, target = images.cuda(), target.cuda()
        images.requires_grad = True
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        cosine_similarity = image_features @ text_features.t()

        ori = images.data.clone().detach()
        if attack:
            if args.attack_type == 'FGSM':
                epsilon = args.epsilon
                pgd_iters = 1
            else:
                epsilon = args.epsilon
                pgd_iters = args.iters
            for iter in range(pgd_iters):
                loss = F.cross_entropy(cosine_similarity, target)
                grad = torch.autograd.grad(loss, images, retain_graph=True)[0]
                images = fgsm_attack(images, epsilon/255, grad).clone().detach()
                if args.attack_type == 'PGD':
                    images.data = ori + torch.clamp(images.data - ori, -args.epsilon/255, args.epsilon/255)
                images.data = ori + torch.clamp(images.data - ori, - ori, 1 - ori)

                images.requires_grad = True
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)
                cosine_similarity = image_features @ text_features.t()
        # end attack

        acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
        tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    VALIDATION = False

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        
        if args.attack_type == 'PGD':

            acc_test = evaluate_lora(args, clip_model, test_loader, dataset, attack=False)
            print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    
            pgd = evaluate_lora(args, clip_model, test_loader, dataset, attack=True)
            print("**** Final test accuracy PGD: {:.2f}. ****\n".format(pgd))
    
        elif args.attack_type == 'FGSM':
            fgsm = evaluate_lora(args, clip_model, test_loader, dataset, attack=True)
            print("**** Final test accuracy FGSM: {:.2f}. ****\n".format(fgsm))
        
        return

    mark_only_lora_as_trainable(clip_model, 'none')
    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        delta = None
        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()

            # tst = torch.ones_like(images[0]).cuda() * 10 /255
            # print(torch.norm(tst))
            # print(images[0].shape)

            if delta is None:
                delta = torch.zeros_like(images[0]).cuda()
                delta.requires_grad = True

            repeated = delta.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
            cloned_images = images.clone()
            images = images + repeated

            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)

            for j in range(args.m):

                delta_grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                if count_iters < 300:
                    step_size = 1 / (torch.linalg.norm(delta) + 1e-30)
                    step_size += (1 - step_size) / (300 - count_iters)
                else:
                    step_size = 1
                step_size *= 0.05  # 0.05
                delta = delta + step_size * delta_grad
                delta = torch.clip(delta, -args.epsilon_train / 255, args.epsilon_train / 255)
                delta = torch.clip(delta, torch.min(-cloned_images), torch.max(1 - cloned_images))

                delta = delta.clone().detach()
                delta.requires_grad = True
                repeated = delta.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
                images = cloned_images + repeated
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
                cosine_similarity = logit_scale * image_features @ text_features.t()
                loss = F.cross_entropy(cosine_similarity, target)

                # delta = None
            

            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward(retain_graph=True)


            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))

        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset, attack=False)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))

    if args.attack_type == 'PGD':

        acc_test = evaluate_lora(args, clip_model, test_loader, dataset, attack=False)
        print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))

        pgd = evaluate_lora(args, clip_model, test_loader, dataset, attack=True)
        print("**** Final test accuracy PGD: {:.2f}. ****\n".format(pgd))

    elif args.attack_type == 'FGSM':
        fgsm = evaluate_lora(args, clip_model, test_loader, dataset, attack=True)
        print("**** Final test accuracy FGSM: {:.2f}. ****\n".format(fgsm))


    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return



