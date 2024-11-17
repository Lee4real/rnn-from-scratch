import time
import torch

# Function to train a model and collect metrics
def train_model(model, train_loader, test_loader, optimizer, loss_fn, n_epochs=5, model_name="Model"):
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses, test_losses = [], []
    train_acc, test_acc = [] , []
    train_times = []

    print(f"\nTraining {model_name}...\n")
    for epoch in range(n_epochs):
        model.train()
        start_time = time.time()

        running_loss, correct_preds, total_preds = 0.0, 0, 0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_preds / total_preds * 100
        train_acc.append(train_accuracy)
        train_times.append(time.time() - start_time)

        model.eval()
        running_loss, correct_preds, total_preds = 0.0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        avg_test_loss = running_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracy = correct_preds / total_preds * 100
        test_acc.append(test_accuracy)

        print(
            f"Epoch {epoch+1}/{n_epochs} - {model_name}\n"
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%\n"
            f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n"
        )

    return train_losses, test_losses, train_acc, test_acc, train_times
