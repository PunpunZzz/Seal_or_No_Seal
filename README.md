# Seal or No Seal

This is a personal project done to experiment with the EfficientNetB0 model
from Keras. The goal was to learn how to train a model on a set of images 
to recognize seals and sea lions among other marine species. The target 
success rate was at least 99% because aiming higher would have been 
difficult due to the limited size of the dataset

## Features

- **Creation of a checkpoint model:** The first part of the training gives a
raw model
- **Creation of a final model:** The second part gives the fine-tuned model
- **Test on a given picture:** Run the program with a picture's link

## Tech used

- **Code:** Python, tensorflow/keras 

## Project Structure

```
ChessWebGame/
├── data/                       # Data to train on
│   ├── No_Seal/                # class 0 
│   └── Seal/                   # class 1
├── best_model_ckp.keras        # raw training      
├── best_model_final.keras      # fine-tuned
├── README.md                   
├── requirements.txt            # Dependencies
├── Seal_or_NoSeal              # Main program to run with your pic
└── training.py                 # To creat/recreat models
```

### Setup

1. Download dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (optional since models are included) Create models:
   ```bash
   python3 training.py
   ```

3. Run program:
   ```bash
   python3 Seal_or_NoSeal.py
   ```

4. Give path to the picture when prompted:
   ```bash
   Give me an image to guess: [include full path]
   ```

5. Get answer:
   ```bash
   In this game of Seal or No Seal, today's winner is....[answer]!!!
   ```
<br>
<div align="center">
  <table border="0" cellspacing="0" cellpadding="0">
    <tr>
      <td align="center" style="border: none; padding: 0;">
        <img src="https://github.com/user-attachments/assets/f1d9df65-71af-438a-bed7-719384a8392f" alt="Seal" width="300" style="display: block;"><br>
        <b>Seal</b>
      </td>
      <td align="center" style="border: none; padding: 0;">
        <img src="https://github.com/user-attachments/assets/3863e028-0f40-4255-aa0b-cdc2a232dd72" alt="No Seal" width="300" style="display: block;"><br>
        <b>No Seal</b>
      </td>
    </tr>
  </table>
</div>

## Limitations

I didn't get any wrong answers during testing, but the results prove the 
accuracy isn't 100%. To get even closer to perfection, you would primarily 
need a much larger dataset to train on
