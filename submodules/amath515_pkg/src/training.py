###############################
# Imports # Imports # Imports #
###############################

import torch
import os
from amath515_pkg.src.helpers import get_config

#####################################
# Generic Helpers # Generic Helpers #
#####################################

def get_optimizer(optimizer_str, parameters, lr, betas=(0.9,0.999)):
    if optimizer_str == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas)
    elif optimizer_str == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=betas)
    elif optimizer_str == 'RMSprop':
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif optimizer_str == 'Adagrad':
        optimizer = torch.optim.Adagrad(parameters, lr=lr)
    else:
        raise Exception(f"Optimizer \"{optimizer_str}\" not valid, use one of \"Adam\", \"AdamW\", \"RMSprop\" or \"Adagrad\"")
    return optimizer

def get_scheduler(optimizer, scheduler_str):
    if scheduler_str == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)
    elif scheduler_str == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=0, last_epoch=-1)
    else:
        raise Exception(f"Scheduler \"{scheduler_str}\" not valid, use one of \"StepLR\" or \"CosineAnnealingLR\"")
    return scheduler

#############################
# CIFAR10 CNN # CIFAR10 CNN #
#############################

def cf10_train_loop(dataloader, model, loss_fn, optimizer, scheduler, batch_size=None):
    # Load config file and values
    config = get_config()
    if batch_size is None:
        batch_size = config["CIFAR10_batch_size"]

    num_batches = len(dataloader) # Total number of batches
    size = 0 # Calculate size of dataset
    train_loss_sum = 0 # Running training loss
    correct = 0 # Running number of correctly identified images

    # Iterate over entire dataset
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        y = torch.nn.functional.one_hot(y, num_classes=10)
        y = y.to(torch.float64)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute statistics
        train_loss_sum += loss
        correct += int((pred.argmax(1) == torch.argmax(y, dim=1)).type(torch.float).sum().item())
        size += X.shape[0]

        # Prints
        loss = loss.item()
        print(f"loss: {loss:>7f} [batch {batch+1} of {num_batches}]")
    scheduler.step()

    # Print results
    train_loss_avg = train_loss_sum / num_batches
    correct_pct = 100. * correct / size
    print()
    print(f"Train error:")
    print(f"  Correct:  {correct} of {size}")
    print(f"  Accuracy: {(correct_pct):>0.2f}%")
    print(f"  Avg loss: {train_loss_avg:>0.6f}")
    return (correct_pct, train_loss_avg)

def cf10_test_loop(dataloader, model, loss_fn):
    size = 0 # Calculate size of dataset
    num_batches = len(dataloader) # Total number of batches
    test_loss_sum = 0 # Running testing loss
    correct = 0 # Running number of correctly identified images

    # Iterate over dataset and infer results using the model
    with torch.no_grad():
        for X, y in dataloader:
            # Evaluate model
            pred = model(X)
            y = torch.nn.functional.one_hot(y, num_classes=10)
            y = y.to(torch.float64)

            # Compute statistics
            test_loss_sum += loss_fn(pred, y).item()
            correct += int((pred.argmax(1) == torch.argmax(y, dim=1)).type(torch.float).sum().item())
            size += X.shape[0]

    # Print results
    test_loss_avg = test_loss_sum / num_batches
    correct_pct = 100.* correct / size
    print(f"Test error:")
    print(f"  Correct:  {correct} of {size}")
    print(f"  Accuracy: {(correct_pct):>0.2f}%")
    print(f"  Avg loss: {test_loss_avg:>0.6f}")
    return (correct_pct, test_loss_avg)


###################
# minGPT # minGPT #
###################

def min_gpt_batch_end_callback(model, mingpt_config, train_dataset, trainer):
    # Load config file and values
    pkg_config = get_config()

    if trainer.iter_num % 100 == 0:
        print(f"Iteration {trainer.iter_num}, lr {trainer.lr:0.6f}\n---------------------------------")
        # evaluate both the train and test score
        model.eval()
        if False: # this just clutters things
            with torch.no_grad():
                # sample from the model...
                context = "O God, O God!"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
        # update scheduler
        trainer.scheduler.step()
        # revert model to training mode
        model.train()

    elif trainer.iter_num % 10 == 0:
        #print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        print(f"loss: {trainer.loss:>7f} [iter {trainer.iter_num} of {mingpt_config.trainer.max_iters}]")

    elif trainer.iter_num % 100 == 99:
            print()

    if trainer.iter_num+1 == mingpt_config.trainer.max_iters:
        # save the last model
        ckpt_path = os.path.join(mingpt_config.system.work_dir, pkg_config['mingpt_saved_filename'])
        print(f"Saving model to \"{ckpt_path}\"")
        torch.save(model.state_dict(), ckpt_path)