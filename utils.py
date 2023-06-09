import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import metrics
import os
from main_model import EMA

def train(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername="", n_epochs=50, patience=7):
    optimizer = Adam(model.parameters(), lr=config["lr"])
    #ema = EMA(0.9)
    #ema.register(model)
    patience_count = 0
    prev_loss = 100_000

    train_losses = []
    val_losses = []

    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"
        
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=150, gamma=.1, verbose=True
    )
    
    best_valid_loss = 1e10
    
    # for epoch_no in range(config["epochs"]):
    for epoch_no in range(n_epochs):
        avg_loss = 0
        model.train()
        
        with tqdm(train_loader) as it:
            for batch_no, (noisy_batch, clean_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                optimizer.zero_grad()
                #print("clean batch size: ", clean_batch.size())
                #print("noisy batch size: ", noisy_batch.size())
                loss = model(clean_batch, noisy_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()
                avg_loss += loss.item()
                
                #ema.update(model)
                
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss (L1)": avg_loss / batch_no,
                        # "avg_epoch_snr": avg_snr / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )

                if batch_no == it.total:
                    train_losses.append(avg_loss / batch_no)
                    # with torch.no_grad():
                    #     predicted_noise = model.retrieve_noise(clean_batch, noisy_batch).cpu().numpy()
                    #     actual_noise = (noisy_batch - clean_batch).cpu().numpy()
                    #     print("predicted noise. min: ", np.min(predicted_noise), ". max: ", np.max(predicted_noise))
                    #     print("actual noise. min: ", np.min(actual_noise), ". max: ", np.max(actual_noise))                         
            
            lr_scheduler.step()
            
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            denoised_snr = 0
            noisy_snr = 0

            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        loss = model(clean_batch, noisy_batch)
                        avg_loss_valid += loss.item()

                        denoised_batch = model.denoise_signal(clean_batch, noisy_batch)
                        
                        denoised_snr += metrics.SNR(denoised_batch, clean_batch)
                        noisy_snr += metrics.SNR(noisy_batch, clean_batch)

                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "denoised_snr": denoised_snr / batch_no,
                                "noisy_snr":  noisy_snr / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )

                        if batch_no == it.total:  
                            val_losses.append(avg_loss_valid / batch_no)
            
            if val_losses[-1] >= prev_loss:
                patience_count += 1
            else:
                patience_count = 0
                prev_loss = val_losses[-1]
            
            if patience_count == patience:
                break
                    
            if best_valid_loss > avg_loss_valid/batch_no:
                best_valid_loss = avg_loss_valid/batch_no
                print("\n best loss is updated to ",avg_loss_valid / batch_no,"at", epoch_no,)
                
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
    
    torch.save(model.state_dict(), final_path)
    np.save(os.path.join(foldername, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(foldername, "val_losses.npy"), np.array(val_losses))
    
def evaluate(model, test_loader, shots, device, foldername=""):
    ssd_total = 0
    mad_total = 0
    # prd_total = 0
    # cos_sim_total = 0
    psnr_total = 0
    gfc_total = 0
    pearson_total = 0
    # snr_noise = 0
    # snr_recon = 0
    # snr_improvement = 0
    eval_points = 0
    
    restored_sig = []
    with tqdm(test_loader) as it:
        for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
            clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
            
            if shots > 1:
                output = 0
                for i in range(shots):
                    output+=model.denoising(noisy_batch)
                output /= shots
            else:
                output = model.denoising(noisy_batch) #B,1,L
            clean_batch = clean_batch.permute(0, 2, 1)
            noisy_batch = noisy_batch.permute(0, 2, 1)
            output = output.permute(0, 2, 1) #B,L,1
            out_numpy = output.cpu().detach().numpy()
            clean_numpy = clean_batch.cpu().detach().numpy()
            noisy_numpy = noisy_batch.cpu().detach().numpy()
            
            
            eval_points += len(output)
            print("eval points: ", eval_points)
            ssd_total += np.sum(metrics.SSD(clean_numpy, out_numpy))
            mad_total += np.sum(metrics.MAD(clean_numpy, out_numpy))
            # prd_total += np.sum(metrics.PRD(clean_numpy, out_numpy))
            # cos_sim_total += np.sum(metrics.COS_SIM(clean_numpy, out_numpy))
            # snr_noise += np.sum(metrics.SNR(clean_numpy, noisy_numpy))
            # snr_recon += np.sum(metrics.SNR(clean_numpy, out_numpy))
            # snr_improvement += np.sum(metrics.SNR_improvement(noisy_numpy, out_numpy, clean_numpy))
            psnr_total += np.sum(metrics.PSNR(clean_numpy, out_numpy))
            gfc_total += np.sum(metrics.computeGFC(clean_numpy, out_numpy))
            pearson_total += np.sum(metrics.computePearson(clean_numpy, out_numpy))
            restored_sig.append(out_numpy)
            
            it.set_postfix(
                ordered_dict={
                    "ssd_total": ssd_total/eval_points,
                    "mad_total": mad_total/eval_points,
                    # "prd_total": prd_total/eval_points,
                    # "cos_sim_total": cos_sim_total/eval_points,
                    # "snr_in": snr_noise/eval_points,
                    # "snr_out": snr_recon/eval_points,
                    "psnr_total": psnr_total / eval_points,
                    "gfc_total": gfc_total / eval_points,
                    "pearson_total": pearson_total / eval_points,
                    # "snr_improve": snr_improvement/eval_points,
                },
                refresh=True,
            )
    
    restored_sig = np.concatenate(restored_sig)
    
    #np.save(foldername + '/denoised.npy', restored_sig)
    
    print("ssd_total: ",ssd_total/eval_points)
    print("mad_total: ", mad_total/eval_points,)
    # print("prd_total: ", prd_total/eval_points,)
    #print("cos_sim_total: ", cos_sim_total/eval_points,)
    # print("snr_in: ", snr_noise/eval_points,)
    # print("snr_out: ", snr_recon/eval_points,)
    # print("snr_improve: ", snr_improvement/eval_points,)
    print("psnr_total: ", psnr_total / eval_points, )
    print("gfc_total: ", gfc_total / eval_points, )
    print("pearson_total: ", pearson_total / eval_points, )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    