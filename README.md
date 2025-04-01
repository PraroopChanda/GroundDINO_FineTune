# GroundingDINO Fine-Tune

This repo contains fine-tuned version of GroundingDINO for custom object detection tasks (KITTI).
---
Some Sample Results Without Fine Tuning

### 🖼️ Example 1 Without Fine Tuning
<p align="center">
  <img src="annotated_image_1.jpg" alt="Dog and Person" width="22%" />
  <img src="annotated_image_2.jpg" alt="Dog Person and Glass" width="22%" />
  <img src="annotated_image_3.jpg" alt="Dog Person and Chair" width="22%" />
  <img src="annotated_image_4.jpg" alt="Dog and Bag" width="22%" />
</p>

<p align="center">
  <b>Figure:</b> From left to right (Input Promt) — Dog and Person, Dog Person and Glass, Dog Person and Chair, Dog and Bag.
</p>


### 🖼️ Example 2 Fine Tuning on Kitti Dataset
<p align="center">
  <img src="annotatated_Kitti.jpg" alt="Car" width="45%" />
  <img src="annotatated_Kitti_person.jpg" alt="Person" width="45%" />
</p>

<p align="center">
  <b>Figure:</b> From left to right (Input Promt) — Car, Person.
</p>




## Model Info

- Base: GroundingDINO
- Fine Tuned on: KITTI Dataset
- Detection Loss: Hungarian matcher + classification + L1/IoU
- GPU: NVIDIA A100
  


The code is build upon [ECCV 2024 Official implementation of the paper](https://github.com/IDEA-Research/GroundingDINO) "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"
