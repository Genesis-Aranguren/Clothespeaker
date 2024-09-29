# Librerías
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BlipForConditionalGeneration, get_scheduler
import torch

# Configuración
modelName = "Salesforce/blip-image-captioning-base"
lr = 1e-5
batchSize = 2
epoch = 20

# Carga del modelo, procesador y optimizador
processor = AutoProcessor.from_pretrained(modelName)
model = BlipForConditionalGeneration.from_pretrained(modelName)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Preprocesamiento del conjunto de datos
class ClothesDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding

dataset = load_dataset("Luna288/image-captioning-FACAD-small", split="train")
train = ClothesDataset(dataset, processor)
train = DataLoader(train, shuffle=True, batch_size=batchSize)

# Configuración del scheduler
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch * len(train)
)

# Entrenamiento
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

for e in range(epoch):
  print("Epoch:", e)
  for idx, batch in enumerate(train):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)
    outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)

    loss = outputs.loss
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

# Guardar el modelo y procesador
model.save_pretrained("./blip-clothes-image-captioning-model")
processor.save_pretrained("./blip-clothes-image-captioning-processor")
