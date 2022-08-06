import json
import os

import torch
from transformers import LEDForConditionalGeneration

from hf_train_bart import Bart

OUTPUT_MODELS = [
    "epoch_11",
    "epoch_13",
    "epoch_18",
    "epoch_6",
    "epoch_2",
    "epoch_3",
    "epoch_0",
    "epoch_1"
]

if __name__ == "__main__":
    model = Bart()
    file_path = model.args.test_file_path
    dev_examples = model.load_examples_for_predict(file_path)
    dev_features = model.convert_examples_to_features_for_predict(dev_examples)

    for epoch_name in OUTPUT_MODELS:
        model.generator = LEDForConditionalGeneration.from_pretrained(os.path.join(model.args.load_path, "outputs", epoch_name))
        model.generator.to(model.device)

        print("[INFO] Predicting for file_path:", file_path)
        pred, pred_kp = model.predict((dev_examples, dev_features), predict=True)

        if not os.path.exists(model.args.output_dir):
            os.makedirs(model.args.output_dir)

        with open(os.path.join(model.args.output_dir, f"predict.pred.summary.{epoch_name}.json"), "w") as fout:
            json.dump(pred, fout, indent=4)

        with open(os.path.join(model.args.output_dir, f"predict.predkp.summary.{epoch_name}.json"), "w") as fout:
            json.dump(pred_kp, fout, indent=4)
