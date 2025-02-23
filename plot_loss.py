import wandb
import matplotlib.pyplot as plt

# wandb.init(project="your_project_name")  # Use your WandB project
history = wandb.Api().runs(path="ralucam-sandu/U-Net-Face")

train_losses = []
val_losses = []
epochs = []

for run in history:
    train_losses.append(run.summary.get("train_loss"))
    val_losses.append(run.summary.get("val_total_loss"))
    epochs.append(run.summary.get("epoch"))

plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()
