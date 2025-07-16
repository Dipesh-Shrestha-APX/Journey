import csv

def combine_csv_remove_header(csv1_path, csv2_path, output_path):
    with open(csv1_path, 'r', newline='', encoding='utf-8') as f1, \
         open(csv2_path, 'r', newline='', encoding='utf-8') as f2, \
         open(output_path, 'w', newline='', encoding='utf-8') as fout:

        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        writer = csv.writer(fout)

        # Read all from first CSV (including its header)
        for row in reader1:
            writer.writerow(row)

        # Skip header of second CSV
        next(reader2, None)

        # Write remaining rows from second CSV
        for row in reader2:
            writer.writerow(row)

    print(f"Combined CSV saved to: {output_path}")

# Example usage:
csv1 = r"D:\Journey\Learnings\CSVs\mnist\fashion-mnist_test.csv"
csv2 = r"D:\Journey\Learnings\CSVs\mnist\fashion-mnist_train.csv"
output = r"D:\Journey\Learnings\CSVs\mnist\fashion-mnist_complete.csv"

combine_csv_remove_header(csv1, csv2, output)
