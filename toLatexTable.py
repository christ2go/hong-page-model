"""
Generates a LaTex Table from the output CSV
"""

import csv
with open("output/output.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        print("\\midrule")
        print(row[0] + " & random & " + row[1] + " ("+row[2]+")" + " & " + str(round(float(row[1])-float(row[3]),2)) + "\\\\")
        print("\t & highest-ability &  " + row[3] + " ("+row[4]+")" + "\\\\")
    print("\\bottomrule")