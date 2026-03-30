import sys
import torch
from PIL import Image
from torchvision import transforms
from src.models.petsv4 import PetsCNNv4

CLASSES = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound",
    "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair",
    "Chihuahua", "Egyptian Mau", "English Cocker Spaniel", "English Setter",
    "German Shorthaired", "Great Pyrenees", "Havanese", "Japanese Chin",
    "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard",
    "Samoyed", "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx",
    "Staffordshire Bull Terrier", "Wheaten Terrier", "Yorkshire Terrier"
]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def classify(image_path: str, weights_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PetsCNNv4(num_classes=37).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top5 = torch.topk(probs, 5)

    print(f"\nObrazek: {image_path}")
    print("-" * 35)
    for prob, idx in zip(top5.values, top5.indices):
        print(f"  {CLASSES[idx.item()]:<35} {prob.item() * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Użycie: python classify.py <ścieżka_do_zdjęcia> <ścieżka_do_wag>")
        print("Przykład: python classify.py kot.jpg PetsCNNv4_best.pth")
        sys.exit(1)

    classify(image_path=sys.argv[1], weights_path=sys.argv[2])
