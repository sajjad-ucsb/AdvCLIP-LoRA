import os

# Directory containing your .txt files
input_dir = 'vit32/log'  # Change this to your directory path
output_file = 'results-2/image-vit32-log-pgd.txt'

results = []

for filename in os.listdir(input_dir):
    if filename.endswith('.log'):
        file_path = os.path.join(input_dir, filename)
        final_accuracy = None
        with open(file_path, 'r') as f:
            for line in f:
                if "Final test accuracy PGD:" in line:
                # if "Final test accuracy:" in line:
                    final_accuracy = line.strip()
        if final_accuracy:
            results.append(f"{filename}: {final_accuracy}")

# Write results to the output file
with open(output_file, 'w') as out_f:
    for result in results:
        out_f.write(result + '\n')

print(f"Results written to {output_file}")




# def extract_final_accuracy():
#     input_dir = 'vit16/log'
#     output_file = 'results/vit16-log.txt'
    
#     with open(output_file, 'w') as out_f:
#         for filename in os.listdir(input_dir):
#             if filename.endswith('.log'):
#                 with open(filename, 'r') as in_f:
#                     accuracy = None
#                     for line in in_f:
#                         if "Final test accuracy:" in line:
#                             # Extract the accuracy value
#                             accuracy = line.split("Final test accuracy:")[1].strip()
                    
#                     if accuracy is not None:
#                         out_f.write(f"{filename}: {accuracy}\n")
#                         print(f"Processed {filename}: {accuracy}")
#                     else:
#                         print(f"No accuracy found in {filename}")

#     print(f"\nAll results written to {output_file}")

# if __name__ == "__main__":
#     extract_final_accuracy()