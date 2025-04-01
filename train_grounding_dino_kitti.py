import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from groundingdino.util import box_ops
import torch.nn.functional as F
from groundingdino.util.misc import nested_tensor_from_tensor_list
from torch.utils.data import Subset
from groundingdino.custom_loss.matcher import HungarianMatcher
from groundingdino.custom_loss.criterion import SetCriterion
from torch.utils.tensorboard import SummaryWriter




# --- Config ---
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"
DATA_ROOT = "/scratch/user/praroop27/kitti_dataset/training/image_2"
ANNOTATIONS_PATH = "/scratch/user/praroop27/kitti_dataset/instances_train_kitti.json"
BATCH_SIZE = 6
EPOCHS = 50
LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# def collate_fn_with_padding(batch):
#     images, targets = list(zip(*batch))
#     images = [img for img in images]  # still tensors
#     return nested_tensor_from_tensor_list(images), targets

CATEGORY_ID_TO_PHRASE = {
    1: "a car",
    2: "a pedestrian",
    3: "a cyclist"
}

writer = SummaryWriter(log_dir="/scratch/user/praroop27/GroundingDINO/runs/kitti_gdino")

# --- Load COCO-format KITTI dataset ---
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return images, targets

dataset = CocoDetection(root=DATA_ROOT, annFile=ANNOTATIONS_PATH, transform=transform)
subset_indices = list(range(1000))
dataset = Subset(dataset, subset_indices)
#dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# --- Load Grounding DINO ---
args = SLConfig.fromfile(CONFIG_PATH)
args.device = DEVICE
model = build_model(args)
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
model = model.to(DEVICE)
model.train()


# --- Freeze BERT + Backbone ---
for name, param in model.named_parameters():
    if "text_encoder" in name or "backbone" in name:
        param.requires_grad = False

# --- Optimizer ---
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# Hungarian matcher and criterion setup
matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
weight_dict = {
    "loss_ce": 1,
    "loss_bbox": 5,
    "loss_giou": 2,
}
losses = ["labels", "boxes"]

criterion = SetCriterion(
    num_classes=3,  # Car, Pedestrian, Cyclist
    matcher=matcher,
    weight_dict=weight_dict,
    eos_coef=0.1,
    losses=losses
).to(DEVICE)


# --- Dummy prompt (will use real class labels later) ---
#prompt = "a car . a pedestrian . a cyclist"

print("This is the device", DEVICE)
torch.autograd.set_detect_anomaly(True)

# --- Training Loop ---
# for epoch in range(EPOCHS):
#     for step, (images, targets) in enumerate(dataloader):
#         #images = [img.to(DEVICE) for img in images]
#         images = torch.stack([img.to(DEVICE) for img in images])
#         optimizer.zero_grad()

#         # Forward pass with dummy prompt
#         outputs = model(images, captions=[prompt] * len(images))

#         # Use only objectness logits and boxes
#         logits = outputs["pred_logits"]
#         boxes = outputs["pred_boxes"]

#         # Dummy loss (simplified version — real matcher comes next)
#         loss = logits.mean() + boxes.mean()

#         loss.backward()
#         optimizer.step()

#         if step % 10 == 0:
#             print(f"[Epoch {epoch}] Step {step} | Loss: {loss.item():.4f}")



# for epoch in range(EPOCHS):
#     for step, (samples, targets) in enumerate(dataloader):
#         samples = samples.to(DEVICE)  # NestedTensor with batched padded input
#         optimizer.zero_grad()

#         # outputs = model(samples, captions=[prompt] * samples.tensors.size(0))

#         # logits = outputs["pred_logits"]
#         # boxes = outputs["pred_boxes"]

#         # #loss = logits.mean() + boxes.mean()
#         # loss = torch.nan_to_num(logits.mean(), nan=0.0, posinf=1e2, neginf=-1e2) + \
#         # torch.nan_to_num(boxes.mean(), nan=0.0, posinf=1e2, neginf=-1e2)
#         outputs = model(samples, captions=[prompt] * samples.tensors.size(0))



#         loss.backward()
#         optimizer.step()

#         if step % 10 == 0:
#             print(f"[Epoch {epoch}] Step {step} | Loss: {loss.item():.4f}")

# for epoch in range(EPOCHS):
#     for step, (images, targets) in enumerate(dataloader):
#         optimizer.zero_grad()

#         # Convert image list to nested tensor (handles padding)
#         samples = nested_tensor_from_tensor_list(images).to(DEVICE)

#         # Use fixed caption for all (can be dynamic later)
#         prompt = "a car . a pedestrian . a cyclist"
#         outputs = model(samples, captions=[prompt] * samples.tensors.size(0))

#         # Use model’s internal loss dictionary
#         loss_dict = outputs["loss_dict"]
#         loss = sum(loss_dict[k] for k in loss_dict if isinstance(loss_dict[k], torch.Tensor))

#         loss.backward()
#         optimizer.step()

#         if step % 10 == 0:
#             losses_str = ", ".join(f"{k}: {v.item():.2f}" for k, v in loss_dict.items() if isinstance(v, torch.Tensor))
#             print(f"[Epoch {epoch}] Step {step} | Total Loss: {loss.item():.4f} | {losses_str}")

# for epoch in range(EPOCHS):
#     model.train()
#     for step, (images, targets) in enumerate(dataloader):
#         optimizer.zero_grad()

#         # Pad images to same size
#         samples = nested_tensor_from_tensor_list(images).to(DEVICE)

#         # Inference with a fixed prompt
#         prompt = "a car . a pedestrian . a cyclist"

#         CATEGORY_ID_TO_PHRASE = {
#             1: "a car",
#             2: "a pedestrian",
#             3: "a cyclist"
#         }


#         # Dynamically build a prompt per image from GT annotations
#         captions = []
#         for anns in targets:
#             labels = [ann["category_id"] for ann in anns]
#             label_names = []
#             for cid in labels:
#                 if cid == 1:
#                     label_names.append("a car")
#                 elif cid == 2:
#                     label_names.append("a pedestrian")
#                 elif cid == 3:
#                     label_names.append("a cyclist")

#             # # Remove duplicates and empty
#             # label_names = sorted(set(label_names))
#             # if not label_names:
#             #     label_names = ["a car"]  # fallback prompt

#             # caption = " . ".join(label_names)
#             # captions.append(caption)

#             caption = label_names[0] if label_names else "a car"  # single object phrase
#             captions.append(caption)            


#         #outputs = model(samples, captions=[prompt] * samples.tensors.size(0))
#         # Manually simplify captions to just one string (e.g., all use 'a car')
#         captions = ['a car .'] * samples.tensors.size(0)
#         outputs = model(samples, captions=captions)

#        # outputs = model(samples, captions=caption, token_spans=None)

#         # Convert COCO targets to format expected by loss
#         new_targets = []
#         for t in targets:
#             boxes = torch.tensor([ann["bbox"] for ann in t], dtype=torch.float32)
#             boxes[:, 2:] += boxes[:, :2]  # [x, y, w, h] → [x1, y1, x2, y2]
#             labels = torch.tensor([ann["category_id"] - 1 for ann in t], dtype=torch.long)  # 0-based indexing

#             new_targets.append({
#                 "boxes": boxes.to(DEVICE),
#                 "labels": labels.to(DEVICE),
#             })

#         # Compute detection losses
#         loss_dict = criterion(outputs, new_targets)
#         loss = sum(loss_dict[k] for k in loss_dict.keys() if k in weight_dict)

#         # Backprop
#         loss.backward()
#         optimizer.step()

#         if step % 10 == 0:
#             print("Captions:", captions)
#         # Logging
#         if step % 10 == 0:
#             for k, v in loss_dict.items():
#                 if isinstance(v, torch.Tensor):
#                     writer.add_scalar(f"Loss/{k}", v.item(), epoch * len(dataloader) + step)
#             writer.add_scalar("Loss/total", loss.item(), epoch * len(dataloader) + step)

# --- TRAINING LOOP ---
for epoch in range(EPOCHS):
    for step, (images, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        samples = nested_tensor_from_tensor_list(images).to(DEVICE)

        # --- Dynamic captions ---
        captions = []
        for anns in targets:
            labels = [ann["category_id"] for ann in anns]
            phrases = [CATEGORY_ID_TO_PHRASE[cid] for cid in labels if cid in CATEGORY_ID_TO_PHRASE]
            if phrases:
                caption = " . ".join(sorted(set(phrases))) + " ."
            else:
                caption = "a car ."
            captions.append(caption)

        print(caption)    

        outputs = model(samples, captions=captions)

        # --- Convert targets ---
        new_targets = []
        for t in targets:
            boxes = torch.tensor([ann["bbox"] for ann in t], dtype=torch.float32)
            boxes[:, 2:] += boxes[:, :2]  # [x, y, w, h] → [x1, y1, x2, y2]
            labels = torch.tensor([ann["category_id"] - 1 for ann in t], dtype=torch.long)
            new_targets.append({"boxes": boxes.to(DEVICE), "labels": labels.to(DEVICE)})

        # --- Loss ---
        loss_dict = criterion(outputs, new_targets)
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        loss.backward()
        optimizer.step()

        # --- Logging ---
        if step % 10 == 0:
            losses_str = ", ".join(f"{k}: {v.item():.2f}" for k, v in loss_dict.items())
            print(f"[Epoch {epoch}] Step {step} | Total Loss: {loss.item():.4f} | {losses_str}")
            for k, v in loss_dict.items():
                writer.add_scalar(f"Loss/{k}", v.item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/total", loss.item(), epoch * len(dataloader) + step)
    if epoch % 5 == 0:
        checkpoint_dir = "/scratch/user/praroop27/GroundingDINO/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"gdino_epoch{epoch}.pth"))

writer.close()

print("✅ Finetuning complete (basic mode)!")
