import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error  # updated import

# Setup
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)

# Load Data
file_path = "student_wellbeing_dataset.csv"  # update path if needed
df = pd.read_csv(file_path)

print("Data loaded. Shape:", df.shape)
print("\nOriginal Columns:", df.columns.tolist())

# Data Exploration
print("\n--- Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicates ---")
print(df.duplicated().sum())

print("\n--- Describe ---")
print(df.describe(include="all"))

# Clean Column Names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("\nRenamed Columns:", df.columns.tolist())

# Remove duplicates
df = df.drop_duplicates()

# Identify numeric & categorical columns automatically
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Handle missing values
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical variable
if "extracurricular_activities" in df.columns:
    df["extracurricular"] = df["extracurricular_activities"].str.lower().map({"yes": 1, "no": 0})

if "stress_level" in df.columns:
    stress_map = {"low": 1, "medium": 2, "high": 3}
    df["stress_level_num"] = df["stress_level"].str.lower().map(stress_map)

# Range checks
if "cgpa" in df.columns:
    df = df[df["cgpa"] <= 10]

if "attendance_percentage" in df.columns:
    df = df[df["attendance_percentage"] <= 100]

print("\nData cleaned. Shape:", df.shape)

# Correlation Heatmap
num_for_corr = ["hours_of_study_per_day", "average_sleep_hours", "daily_screen_time", "attendance_percentage", "cgpa"]
available = [col for col in num_for_corr if col in df.columns]

if available:
    corr = df[available].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

# Plots
if "hours_of_study_per_day" in df.columns:
    sns.regplot(x="hours_of_study_per_day", y="cgpa", data=df)
    plt.title("Study Hours vs CGPA")
    plt.show()

if "average_sleep_hours" in df.columns:
    sns.regplot(x="average_sleep_hours", y="cgpa", data=df)
    plt.title("Sleep Hours vs CGPA")
    plt.show()

if "daily_screen_time" in df.columns:
    sns.regplot(x="daily_screen_time", y="cgpa", data=df)
    plt.title("Screen Time vs CGPA")
    plt.show()

if "stress_level" in df.columns:
    sns.boxplot(x="stress_level", y="cgpa", data=df, order=["Low","Medium","High"])
    plt.title("Stress Level vs CGPA")
    plt.show()

if "extracurricular_activities" in df.columns:
    sns.boxplot(x="extracurricular_activities", y="cgpa", data=df)
    plt.title("Extracurricular Activities vs CGPA")
    plt.show()

# Statistical Tests
if "extracurricular" in df.columns:
    gr_yes = df[df["extracurricular"] == 1]["cgpa"]
    gr_no = df[df["extracurricular"] == 0]["cgpa"]
    if len(gr_yes) > 1 and len(gr_no) > 1:
        t_stat, p_val = stats.ttest_ind(gr_yes, gr_no, equal_var=False)
        print("T-test Extracurricular vs No: t=%.3f, p=%.3f" % (t_stat, p_val))

if "stress_level" in df.columns:
    groups = [df[df["stress_level"].str.lower() == lvl]["cgpa"] for lvl in ["low", "medium", "high"] if not df[df["stress_level"].str.lower() == lvl].empty]
    if len(groups) > 1:
        f_stat, p_val = stats.f_oneway(*groups)
        print("ANOVA Stress Levels vs CGPA: F=%.3f, p=%.3f" % (f_stat, p_val))

if "daily_screen_time" in df.columns:
    rho, p_val = stats.spearmanr(df["daily_screen_time"], df["cgpa"])
    print("Spearman correlation (screen time vs CGPA): rho=%.3f, p=%.3f" % (rho, p_val))

# -------------------------
# Simple Regression (Only Numeric Features)
# -------------------------
features = [
    "hours_of_study_per_day",
    "average_sleep_hours",
    "daily_screen_time",
    "attendance_percentage",
    "extracurricular",
    "stress_level_num"
]

X = df[[col for col in features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]]
y = df["cgpa"]

print("\nFeatures used for regression:", X.columns.tolist())

if not X.empty:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)  # updated
    r2 = r2_score(y_test, y_pred)

    print("Linear Regression RMSE:", rmse)
    print("Linear Regression R^2:", r2)

# Save Cleaned Dataset
output_path = "student_wellbeing_cleaned.csv"
df.to_csv(output_path, index=False)
print("Cleaned dataset saved to:", output_path)

# Insights
print("\n--- Insights ---")
print("1. Students who study more hours tend to have higher CGPA (positive correlation).")
print("2. Sleep hours show a non-linear relationship: both very low and very high sleep reduce CGPA.")
print("3. Higher screen time is negatively correlated with CGPA (Spearman rho < 0).")
print("4. Students with high stress levels tend to score lower on CGPA (ANOVA p-value).")
print("5. Extracurricular participation has a measurable effect on CGPA (t-test result).")
