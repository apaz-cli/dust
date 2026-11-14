# Is the readout worth its cost? It makes the residual-stream draws 1.83x slower (readout_time_l4). Against it: the
# recipe without the readout but with twice the writer, MLP-input and embedding draws (more compute than the
# readout costs). Compare readout_l4 (recipe / +readout). 4 layers, 3000 steps, seeds 42 and 7.
from _common import COPY_TASK, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco"] + LEAKFREE_CV + ["--local-scale", "1", "--direct-scale", "2",
                                                                    "--emb-scale", "8"]
OPTIONS = {
    ".data": {"values": ["tinystories", "copy"],
              "flags": {"tinystories": ["--data-dir", TINYSTORIES],
                        "copy": ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98"]}},
    ".seed": {"values": [42, 7], "flags": "--seed"},
}
METRIC = "val_loss"
GOAL = "minimize"
