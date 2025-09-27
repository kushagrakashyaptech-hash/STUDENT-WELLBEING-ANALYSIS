# Student Wellbeing and Academic Performance Analysis

## Objective
The goal of this project is to explore how lifestyle factors affect students’ academic performance (CGPA). We investigate the impact of study hours, sleep, screen time, attendance, stress levels, and participation in extracurricular activities on student grades.

## Dataset
The dataset contains the following key columns:
- `hours_of_study_per_day` – Average daily study hours  
- `average_sleep_hours` – Average sleep per day  
- `daily_screen_time` – Average daily screen time in hours  
- `attendance_percentage` – Class attendance (%)  
- `cgpa` – Cumulative GPA (scale 0–10)  
- `stress_level` – Self-reported stress (Low/Medium/High)  
- `extracurricular_activities` – Participation in extracurriculars (Yes/No)  

## Steps
1. **Data Cleaning**
   - Removed duplicates and handled missing values (median for numeric, mode for categorical).  
   - Encoded categorical variables numerically (`extracurricular`, `stress_level_num`).  
   - Checked ranges for CGPA and attendance.  

2. **Exploratory Data Analysis (EDA)**
   - Correlation heatmaps between numeric variables.  
   - Scatter plots for study hours, sleep, screen time vs CGPA.  
   - Boxplots for stress level and extracurricular activities vs CGPA.  

3. **Statistical Tests**
   - T-test for CGPA difference between students with and without extracurriculars.  
   - ANOVA for CGPA differences across stress levels.  
   - Spearman correlation between screen time and CGPA.  

4. **Regression Model**
   - Linear regression using numeric features: study hours, sleep, screen time, attendance, stress level, extracurricular participation.  
   - Evaluated using RMSE and R² score.  

5. **Insights**
   - More study hours correlate with higher CGPA.  
   - Extreme sleep patterns (too little or too much) reduce CGPA.  
   - Higher daily screen time is negatively correlated with CGPA.  
   - Higher stress levels correspond to lower CGPA.  
   - Participation in extracurricular activities has a measurable effect on CGPA.  

## Results
- Correlation heatmap and scatter plots visually show relationships.  
- Boxplots reveal differences in CGPA by stress level and extracurricular activities.  
- Regression model achieved reasonable R² and RMSE, confirming the influence of lifestyle factors.  

**Conclusion:** Students’ study habits, sleep, screen time, stress, and extracurricular involvement significantly impact academic performance. This analysis can guide interventions to improve wellbeing and grades.
