"""
Génère un dataset synthétique de messages spam/ham (CSV).
"""
import pandas as pd
import random

random.seed(42)

SPAM = [
    "Congratulations you won a free iPhone click here now",
    "URGENT your account has been compromised verify immediately",
    "Make money fast earn 5000 dollars per day working from home",
    "You are selected for an exclusive offer limited time only",
    "FREE gift card claim your 500 dollar Amazon voucher now",
    "Buy cheap medications online no prescription needed",
    "Hot singles in your area are waiting for you",
    "Your package is on hold click to reschedule delivery",
    "Win a luxury vacation for two reply with your details",
    "Invest in crypto now guaranteed 300 percent return",
    "You have been approved for a loan of 10000 dollars",
    "Lose weight fast with our miracle pill buy now",
    "Your bank account is suspended verify your identity",
    "Click here to claim your prize you are the lucky winner",
    "Limited offer buy one get ten free act now",
    "Dear customer your payment failed update billing info",
    "Earn passive income from home no experience required",
    "Final notice your subscription will expire renew now",
    "You owe back taxes pay immediately to avoid arrest",
    "Cheap software licenses Microsoft Office only 5 dollars",
]

HAM = [
    "Hi can we schedule a meeting for tomorrow at 10am",
    "Please find attached the quarterly report for review",
    "The project deadline is next Friday let me know your progress",
    "Could you review the pull request I submitted yesterday",
    "Let us grab lunch tomorrow if you are free",
    "The conference call is rescheduled to 3pm today",
    "I will be out of office next week back on Monday",
    "Thanks for your help on the presentation it looks great",
    "Can you send me the updated budget spreadsheet",
    "The client approved the proposal we can start Monday",
    "Please remember to submit your timesheet by end of day",
    "The server maintenance is scheduled for Sunday night",
    "I reviewed your code changes they look good to merge",
    "Can we move the standup to 9am instead of 10am",
    "The new hire starts next Monday please prepare access",
    "Your vacation request for next week has been approved",
    "The database backup completed successfully last night",
    "Please update the documentation before the release",
    "The team retrospective is at 4pm in conference room B",
    "Your expense report has been processed and approved",
]

def generate_dataset(n_samples: int = 300, output_path: str = "data/messages.csv"):
    messages, labels = [], []

    for _ in range(n_samples // 2):
        msg = random.choice(SPAM)
        # légère variation pour diversifier
        words = msg.split()
        random.shuffle(words[:3])
        messages.append(" ".join(words))
        labels.append(1)  # spam

    for _ in range(n_samples // 2):
        msg = random.choice(HAM)
        words = msg.split()
        random.shuffle(words[:2])
        messages.append(" ".join(words))
        labels.append(0)  # ham

    combined = list(zip(messages, labels))
    random.shuffle(combined)
    messages, labels = zip(*combined)

    df = pd.DataFrame({"text": messages, "label": labels})
    df.to_csv(output_path, index=False)
    print(f"[generate_data] {len(df)} messages saved → {output_path}")
    print(f"  spam: {df['label'].sum()}  |  ham: {(df['label'] == 0).sum()}")
    return df


if __name__ == "__main__":
    generate_dataset()
