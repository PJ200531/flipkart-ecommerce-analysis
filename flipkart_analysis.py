import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os

# Unzip the dataset (re-doing this here for self-containment of the cell if needed)
zip_file_path = 'Flipkart E-commerce Dataset — 20,000 samples, useful columns..zip'
extracted_dir = 'flipkart_data'

# Check if the directory exists and contains the file before unzipping
if not os.path.exists(extracted_dir) or not os.path.exists(os.path.join(extracted_dir, 'flipkart_com-ecommerce_sample.csv')):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)
    print("Dataset unzipped.")

# Find the CSV file in the extracted directory
csv_files = [f for f in os.listdir(extracted_dir) if f.endswith('.csv')]

if csv_files:
    csv_file_path = os.path.join(extracted_dir, csv_files[0])
    print(f"Found CSV file: {csv_file_path}")
else:
    print("No CSV file found in the extracted directory.")
    csv_file_path = None # Set to None if no CSV is found

# Load the dataset from the extracted CSV file
if csv_file_path:
    df = pd.read_csv(csv_file_path, encoding="utf-8")
    print("Dataset loaded successfully!")
else:
    print("Failed to load dataset as no CSV file was found.")


# Basic cleaning
df.drop_duplicates(inplace=True)

# Fill missing ratings or prices with median values (if they exist)
if "discounted_price" in df.columns:
    df["discounted_price"] = df["discounted_price"].fillna(df["discounted_price"].median())
if "retail_price" in df.columns: # Use retail_price as actual_price seems to be retail_price
    df["retail_price"] = df["retail_price"].fillna(df["retail_price"].median())


# Convert prices to numeric (remove ₹ and commas)
def clean_price(x):
    if isinstance(x, str):
        x = x.replace("₹", "").replace(",", "").strip()
        return float(x) if x.replace('.', '', 1).isdigit() else None
    return x

for col in ["discounted_price", "retail_price"]:
    if col in df.columns:
        df[col] = df[col].apply(clean_price)

# Add discount percentage column
# Using retail_price as actual_price seems to be retail_price based on column names
if "retail_price" in df.columns and "discounted_price" in df.columns:
    df['discount_percent'] = ((df['retail_price'] - df['discounted_price']) / df['retail_price']) * 100

# Drop rows where essential price or discount data is missing after cleaning
df.dropna(subset=['discounted_price', 'retail_price', 'discount_percent'], inplace=True)


# --- Basic Stats ---
avg_discount = df['discount_percent'].mean() if 'discount_percent' in df.columns else None
# Using product_category_tree as category based on column names
top_category = df['product_category_tree'].mode()[0] if 'product_category_tree' in df.columns else None
highest_priced = df.loc[df['discounted_price'].idxmax()]['product_name'] if 'discounted_price' in df.columns and not df['discounted_price'].empty else "N/A"

# --- Summary Report ---
print("\n📊 Flipkart E-Commerce Analysis Report 📊")
print("="*45)
print(f"Total Products Analyzed: {len(df)}")
if top_category:
    print(f"Top Category: {top_category}")
if avg_discount is not None:
    print(f"Average Discount: {avg_discount:.2f}%")
# Using product_rating as ratings based on column names
if 'product_rating' in df.columns and not df['product_rating'].empty:
    # Need to handle "No rating available" and convert to numeric
    df['product_rating_numeric'] = pd.to_numeric(df['product_rating'], errors='coerce')
    avg_rating = df['product_rating_numeric'].mean()
    if not pd.isna(avg_rating):
        print(f"Average Rating: {avg_rating:.2f}⭐")
print(f"Highest Priced Product: {highest_priced}")
print("="*45)

# --- Visuals ---
if 'discounted_price' in df.columns and not df['discounted_price'].empty:
    plt.figure(figsize=(7,4))
    sns.histplot(df['discounted_price'], bins=50, kde=True)
    plt.title("Product Price Distribution")
    plt.xlabel("Price (₹)")
    plt.ylabel("Count")
    plt.show()

if 'discount_percent' in df.columns and not df['discount_percent'].empty:
    plt.figure(figsize=(6,4))
    sns.histplot(df['discount_percent'], bins=40, color='orange')
    plt.title("Discount Percentage Distribution")
    plt.xlabel("Discount %")
    plt.ylabel("Number of Products")
    plt.show()

if 'product_category_tree' in df.columns and not df['product_category_tree'].empty:
    plt.figure(figsize=(8,4))
    df['product_category_tree'].value_counts().head(10).plot(kind='bar', color='green')
    plt.title("Top 10 Product Categories")
    plt.ylabel("Count")
    plt.xlabel("Category")
    plt.show()

if 'product_rating_numeric' in df.columns and not df['product_rating_numeric'].empty:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df['product_rating_numeric'], color='purple')
    plt.title("Rating Distribution")
    plt.xlabel("Ratings")
    plt.show()
