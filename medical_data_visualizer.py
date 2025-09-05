import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data
df = pd.read_csv('medical_examination.csv')

# 2. Add an 'overweight' column
# Calculate BMI: weight (kg) / (height (m))^2
bmi = df['weight'] / (df['height'] / 100)**2
# Add overweight column: 1 if BMI > 25, else 0
df['overweight'] = (bmi > 25).astype(int)

# 3. Normalize data
# cholesterol: 1 (normal) -> 0, >1 (above normal) -> 1
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
# gluc: 1 (normal) -> 0, >1 (above normal) -> 1
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Draw the Categorical Plot
def draw_cat_plot():
    # 5. Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Group and reformat the data to split it by 'cardio'
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    df_cat = df_cat.rename(columns={'size': 'total'})
    
    # 7. Convert the data into long format and create a chart
    # Use sns.catplot() to create the plot
    g = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar')

    # 8. Get the figure for the output
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10. Draw the Heat Map
def draw_heat_map():
    # 11. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Plot the correlation matrix using seaborn's heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        linewidths=.5,
        vmin=-0.16,  # Adjust vmin and vmax to better match the example
        vmax=0.32,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )



    # 16
    fig.savefig('heatmap.png')
    return fig
