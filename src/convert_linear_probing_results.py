"""
Converts a list of pickle files from linear probing to csv complying
with the format of finetuning.
"""

from typing import Any, Dict, List, Tuple

import argparse
import csv
import pickle

from pathlib import Path


# epoch,
# acc,
# esacc_attr0, esacc_attr1, esacc_attr2, esacc_attr3,
# auc,
# esauc_attr0, esauc_attr1, esauc_attr2, esauc_attr3,
# auc_attr0_group0, auc_attr0_group1, auc_attr0_group2,
# auc_attr1_group0, auc_attr1_group1,
# auc_attr2_group0, auc_attr2_group1,
# auc_attr3_group0, auc_attr3_group1, auc_attr3_group2,
# dpd_attr0, dpd_attr1, dpd_attr2, dpd_attr3,
# eod_attr0, eod_attr1, eod_attr2, eod_attr3,
# std_group_disparity_attr0,
# max_group_disparity_attr0,
# std_group_disparity_attr1,
# max_group_disparity_attr1,
# std_group_disparity_attr2,
# max_group_disparity_attr2,
# std_group_disparity_attr3,
# max_group_disparity_attr3,
# path

def convert_data_to_csv(data: List[Tuple[str, Dict[str, Any]]], scale: float = 100.0) -> List[str]:
    csv_content = [
        [
            'epoch',
            'acc',
            'esacc_attr0', 'esacc_attr1', 'esacc_attr2', 'esacc_attr3',
            'auc',
            'esauc_attr0', 'esauc_attr1', 'esauc_attr2', 'esauc_attr3',
            'auc_attr0_group0', 'auc_attr0_group1', 'auc_attr0_group2',
            'auc_attr1_group0', 'auc_attr1_group1',
            'auc_attr2_group0', 'auc_attr2_group1',
            'auc_attr3_group0', 'auc_attr3_group1', 'auc_attr3_group2',
            'dpd_attr0', 'dpd_attr1', 'dpd_attr2', 'dpd_attr3',
            'eod_attr0', 'eod_attr1', 'eod_attr2', 'eod_attr3',
            'std_group_disparity_attr0', 'max_group_disparity_attr0',
            'std_group_disparity_attr1', 'max_group_disparity_attr1',
            'std_group_disparity_attr2', 'max_group_disparity_attr2',
            'std_group_disparity_attr3', 'max_group_disparity_attr3',
            'path'
        ]
    ]

    for file_name, content in data:
        epoch = ""
        acc = content["overall_acc"]

        esacc_attr0 = content["eval_es_acc"][0]
        esacc_attr1 = content["eval_es_acc"][1]
        esacc_attr2 = content["eval_es_acc"][2]
        esacc_attr3 = content["eval_es_acc"][3]

        auc = content["overall_auc"]

        esauc_attr0 = content["eval_es_auc"][0]
        esauc_attr1 = content["eval_es_auc"][1]
        esauc_attr2 = content["eval_es_auc"][2]
        esauc_attr3 = content["eval_es_auc"][3]

        auc_attr0_group0 = content["eval_aucs_by_attrs"][0][0]
        auc_attr0_group1 = content["eval_aucs_by_attrs"][0][1]
        auc_attr0_group2 = content["eval_aucs_by_attrs"][0][2]

        auc_attr1_group0 = content["eval_aucs_by_attrs"][1][0]
        auc_attr1_group1 = content["eval_aucs_by_attrs"][1][1]

        auc_attr2_group0 = content["eval_aucs_by_attrs"][2][0]
        auc_attr2_group1 = content["eval_aucs_by_attrs"][2][1]

        auc_attr3_group0 = content["eval_aucs_by_attrs"][3][0]
        auc_attr3_group1 = content["eval_aucs_by_attrs"][3][1]
        auc_attr3_group2 = content["eval_aucs_by_attrs"][3][2]

        dpd_attr0 = content["eval_dpds"][0]
        dpd_attr1 = content["eval_dpds"][1]
        dpd_attr2 = content["eval_dpds"][2]
        dpd_attr3 = content["eval_dpds"][3]

        eod_attr0 = content["eval_eods"][0]
        eod_attr1 = content["eval_eods"][1]
        eod_attr2 = content["eval_eods"][2]
        eod_attr3 = content["eval_eods"][3]

        std_group_disparity_attr0 = content["between_group_disparity"][0][0]
        max_group_disparity_attr0 = content["between_group_disparity"][0][1]

        std_group_disparity_attr1 = content["between_group_disparity"][1][0]
        max_group_disparity_attr1 = content["between_group_disparity"][1][1]

        std_group_disparity_attr2 = content["between_group_disparity"][2][0]
        max_group_disparity_attr2 = content["between_group_disparity"][2][1]

        std_group_disparity_attr3 = content["between_group_disparity"][3][0]
        max_group_disparity_attr3 = content["between_group_disparity"][3][1]

        path = file_name.replace(".pickle", "")

        csv_line = [
            acc,
            esacc_attr0, esacc_attr1, esacc_attr2, esacc_attr3,
            auc,
            esauc_attr0, esauc_attr1, esauc_attr2, esauc_attr3,
            auc_attr0_group0, auc_attr0_group1, auc_attr0_group2,
            auc_attr1_group0, auc_attr1_group1,
            auc_attr2_group0, auc_attr2_group1,
            auc_attr3_group0, auc_attr3_group1, auc_attr3_group2,
            dpd_attr0, dpd_attr1, dpd_attr2, dpd_attr3,
            eod_attr0, eod_attr1, eod_attr2, eod_attr3,
            std_group_disparity_attr0, max_group_disparity_attr0,
            std_group_disparity_attr1, max_group_disparity_attr1,
            std_group_disparity_attr2, max_group_disparity_attr2,
            std_group_disparity_attr3, max_group_disparity_attr3,
        ]
        csv_line = [value * scale for value in csv_line]
        csv_line = [epoch, *csv_line, path]
        # csv_line.append(path)
        csv_content.append(csv_line)

    return csv_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert output files of linear probing experiments')
    parser.add_argument('--files', type=str, nargs="+", required=True, help='name of files to convert')
    parser.add_argument('--out', type=str, required=True, help='name of target csv file')
    parser.add_argument('--base_dir', type=str, required=True, help='directory containing files')
    parser.add_argument('--rescale', action="store_true", help='rescales scores to lie in [0, 100] instead of [0, 1]')
    parser.add_argument('--verbose', '--v', action="store_true", help='verbose output')
    args = parser.parse_args()

    data = []
    base_dir = Path(args.base_dir)

    print(args.files)

    for file_name in args.files:
        file_path = base_dir / file_name
        with open(file_path, "rb") as f:
            content = pickle.load(f)
            data.append((file_name, content))

    scale = 100 if args.rescale else 1
    converted_data = convert_data_to_csv(data, scale=scale)

    with open(args.out, "w", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        writer.writerows(converted_data)

    if args.verbose:
        print(f"Headers: {converted_data[0]}")
        print(f"Wrote {len(converted_data) - 1} pickle files to {args.out}")



