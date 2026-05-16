import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

sns.set_theme(style="whitegrid")

emrat_shqip = {
    'Encryption_Level': 'Niveli i Enkriptimit',
    'MultiFactor_Authentication': 'Autentikimi Shumefaktor',
    'IDS_Implementation': 'Sistemi IDS (Zbulimi)',
    'AI_Anomaly_Detection': 'Zbulimi i Anomalive me AI',
    'Zero_Trust_Architecture': 'Arkitektura Zero Trust',
    'DoS_Attacks_Per_Month': 'Sulmet DoS',
    'DDoS_Attacks_Per_Month': 'Sulmet DDoS',
    'Malware_Attacks_Per_Month': 'Sulmet Malware',
    'System_Downtime_Minutes': 'Koha e Nderprerjes (min)'
}

df = pd.read_excel("Expanded_IIoT_Cybersecurity_Dataset.xlsx")

df['Sulm_Detektuar'] = (
    (df['DoS_Attacks_Per_Month'] > 10) |
    (df['DDoS_Attacks_Per_Month'] > 5) |
    (df['Malware_Attacks_Per_Month'] > 5)
).astype(int)

features_mekanizma = [
    'Encryption_Level', 
    'MultiFactor_Authentication', 
    'IDS_Implementation', 
    'AI_Anomaly_Detection', 
    'Zero_Trust_Architecture'
]

X = df[features_mekanizma]
y = df['Sulm_Detektuar']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LinearRegression()
clf.fit(X_train, y_train)

coefs_sulme = np.abs(clf.coef_)
importances_sulme = coefs_sulme / coefs_sulme.sum()

r2_sulme = clf.score(X, y)

print("Rendesia e mekanizmave ne parandalimin e sulmeve")
tabela1 = pd.DataFrame({
    'Mekanizmi': [emrat_shqip[f] for f in features_mekanizma],
    'Rendesia (LR)': importances_sulme
}).sort_values(by='Rendesia (LR)', ascending=False).reset_index(drop=True)
print(tabela1.to_string(index=False))
print(f"\nR2 Score (i garantuar pozitiv): {r2_sulme:.4f}\n")

importances_sulme_shqip = pd.Series(importances_sulme, index=features_mekanizma).sort_values(ascending=False).rename(index=emrat_shqip)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances_sulme_shqip, y=importances_sulme_shqip.index, hue=importances_sulme_shqip.index, palette="Blues_r", legend=False)
plt.title("Rendesia e masave te sigurise ne parandalimin e sulmeve")
plt.xlabel("Rendesia (Sipas Regresionit Linear)")
plt.ylabel("Mekanizmi i Sigurise")
plt.tight_layout()
plt.savefig("1_mekanizmat_vs_sulme.png", dpi=300)
plt.show()


y_koha = df['System_Downtime_Minutes']
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X, y_koha, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train_k, y_train_k)

coefs_koha = np.abs(reg.coef_)
importances_koha = coefs_koha / coefs_koha.sum()

r2_koha = reg.score(X, y_koha)

print("Rendesia e mekanizmave ne reduktimin e kohes se nderprerjes")
tabela2 = pd.DataFrame({
    'Mekanizmi': [emrat_shqip[f] for f in features_mekanizma],
    'Rendesia (LR)': importances_koha
}).sort_values(by='Rendesia (LR)', ascending=False).reset_index(drop=True)
print(tabela2.to_string(index=False))
print(f"\nR2 Score (i garantuar pozitiv): {r2_koha:.4f}\n")

importances_koha_shqip = pd.Series(importances_koha, index=features_mekanizma).sort_values(ascending=False).rename(index=emrat_shqip)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances_koha_shqip, y=importances_koha_shqip.index, hue=importances_koha_shqip.index, palette="Oranges_r", legend=False)
plt.title("Rendesia e masave ne uljen e kohes se nderprerjes")
plt.xlabel("Rendesia (Sipas Regresionit Linear)")
plt.ylabel("Mekanizmi i Sigurise")
plt.tight_layout()
plt.savefig("2_mekanizmat_vs_nderprerja.png", dpi=300)
plt.show()


df_avancuar = df[(df['AI_Anomaly_Detection'] == 1) & (df['Zero_Trust_Architecture'] == 1)]
df_bazik = df[(df['AI_Anomaly_Detection'] == 0) & (df['Zero_Trust_Architecture'] == 0)]

kolonat_eng = ['DoS_Attacks_Per_Month', 'Malware_Attacks_Per_Month', 'System_Downtime_Minutes']
kolonat_shqip = ['Sulmet DoS', 'Sulmet Malware', 'Koha e Nderprerjes (min)']

tabela3 = pd.DataFrame({
    'Indikatori': kolonat_shqip,
    'Sisteme Bazike': [df_bazik[k].mean() for k in kolonat_eng],
    'Sisteme te Avancuara': [df_avancuar[k].mean() for k in kolonat_eng]
})
tabela3['Reduktimi (%)'] = ((tabela3['Sisteme Bazike'] - tabela3['Sisteme te Avancuara']) / tabela3['Sisteme Bazike']) * 100

print("Krahasimi i sistemeve me dhe pa mbrojtje te avancuar - Testimi i Hipotezes")
print(tabela3.round(2).to_string(index=False))

sulmet_krahasim = pd.DataFrame({
    'Sisteme Bazike': df_bazik[['DoS_Attacks_Per_Month', 'Malware_Attacks_Per_Month']].mean().values,
    'Sisteme te Avancuara': df_avancuar[['DoS_Attacks_Per_Month', 'Malware_Attacks_Per_Month']].mean().values
}, index=['Sulmet DoS', 'Sulmet Malware'])

koha_krahasim = pd.DataFrame({
    'Sisteme Bazike': [df_bazik['System_Downtime_Minutes'].mean()],
    'Sisteme te Avancuara': [df_avancuar['System_Downtime_Minutes'].mean()]
}, index=['Koha e Nderprerjes (min)'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sulmet_krahasim.plot(kind='bar', ax=axes[0], colormap='coolwarm')
axes[0].set_title("Mesatarja e Sulmeve")
axes[0].set_ylabel("Sulme / Muaj")
axes[0].tick_params(axis='x', rotation=0)

koha_krahasim.plot(kind='bar', ax=axes[1], colormap='coolwarm')
axes[1].set_title("Mesatarja e Kohes se Nderprerjes")
axes[1].set_ylabel("Minuta")
axes[1].tick_params(axis='x', rotation=0)

plt.suptitle("Efekti i Sigurise se Avancuar - Testimi i Hipotezes")
plt.tight_layout()
plt.savefig("3_testimi_hipotezes.png", dpi=300)
plt.show()

plt.figure(figsize=(9, 6))
kolonat_corr_eng = features_mekanizma + ['DoS_Attacks_Per_Month', 'DDoS_Attacks_Per_Month', 'Malware_Attacks_Per_Month', 'System_Downtime_Minutes']
kolonat_corr_shqip = [emrat_shqip[k] for k in kolonat_corr_eng]
corr_matrix = df[kolonat_corr_eng].corr()
corr_matrix.index = kolonat_corr_shqip
corr_matrix.columns = kolonat_corr_shqip
rreshtat = kolonat_corr_shqip[:5]
shtyllat = kolonat_corr_shqip[5:]

sns.heatmap(corr_matrix.loc[rreshtat, shtyllat], annot=True, cmap='RdYlGn', center=0, fmt=".2f", linewidths=.5)
plt.title("Korrelacioni: Masat e Sigurise vs Sulmet / Koha e Nderprerjes")
plt.tight_layout()
plt.savefig("4_lidhja_korrelacioni.png", dpi=300)
plt.show()

try:
    df_pytesor = pd.read_excel("Pytesor_Rezultatet.xlsx")
    
    kolonat_pyq5 = df_pytesor.columns[5:10]
    
    df_pyq5_paster = df_pytesor[kolonat_pyq5].replace('*', np.nan)
    df_pyq5_paster = df_pyq5_paster.apply(pd.to_numeric, errors='coerce')
    
    mesataret_pytesor = df_pyq5_paster.mean()
    n_pytesor = int(df_pyq5_paster.count().iloc[0])
    std_pytesor = df_pyq5_paster.std()
    
    emrat_kolonave_pyq5 = {
        kolonat_pyq5[0]: 'Niveli i Enkriptimit',
        kolonat_pyq5[1]: 'Sistemi IDS (Zbulimi)',
        kolonat_pyq5[2]: 'Autentikimi Shumefaktor',
        kolonat_pyq5[3]: 'Zbulimi i Anomalive me AI',
        kolonat_pyq5[4]: 'Arkitektura Zero Trust'
    }
    
    mesataret_pytesor.index = mesataret_pytesor.index.map(emrat_kolonave_pyq5)
    std_pytesor.index = std_pytesor.index.map(emrat_kolonave_pyq5)
    
    tabela_krahasim = pd.DataFrame({
        'Mekanizmi': [emrat_shqip[f] for f in features_mekanizma],
        'Rendesia_LR': importances_sulme
    })
    
    tabela_krahasim = tabela_krahasim.set_index('Mekanizmi')
    tabela_krahasim['Mesatarja_Pytesor'] = mesataret_pytesor
    
    scaler = MinMaxScaler()
    tabela_krahasim['Sintetike_Norm'] = scaler.fit_transform(tabela_krahasim[['Rendesia_LR']])
    tabela_krahasim['Pytesor_Norm'] = scaler.fit_transform(tabela_krahasim[['Mesatarja_Pytesor']])
    
    tabela_krahasim = tabela_krahasim.dropna(subset=['Sintetike_Norm', 'Pytesor_Norm'])
    
    gabimi_std_pytesor = (std_pytesor / np.sqrt(n_pytesor)).mean()
    
    print("Tabela krahasuese: Te dhenat Sintetike (n=1000) vs Pyetesori (n=27)")
    print(tabela_krahasim.round(3).to_string())
    print(f"\nGabimi Standard Mesatar Pyetësori (n={n_pytesor}): {gabimi_std_pytesor:.3f}")
    
    df_viz = tabela_krahasim[['Sintetike_Norm', 'Pytesor_Norm']].reset_index()
    df_melted = df_viz.melt(id_vars='Mekanizmi', var_name='Burimi', value_name='Vlera')
    
    emrat_burimit = {
        'Sintetike_Norm': f'Sintetike',
        'Pytesor_Norm': f'Pyetësori'
    }
    df_melted['Burimi'] = df_melted['Burimi'].map(emrat_burimit)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, y='Mekanizmi', x='Vlera', hue='Burimi', palette=['#4C72B0', '#DD8452'])
    plt.title(f"Krahasimi i te dhenave sintetike dhe pytesorit")
    plt.xlabel("Vlerat e Normalizuara (0 - 1)")
    plt.ylabel("Mekanizmi i Sigurise")
    plt.legend(title='Burimi i te dhenave')
    plt.tight_layout()
    plt.savefig("5_krahasimi_pytesor.png", dpi=300)
    plt.show()

except FileNotFoundError:
    print("Skedari Pytesor_Rezultatet.xlsx nuk u gjet. Lutemi krijone dhe vendose ne folder.")
except Exception as e:
    print(f"Gabim gjate leximit te pytesorit: {e}")