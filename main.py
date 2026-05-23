import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler

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

sum_neg = len(y_train) - sum(y_train)
sum_pos = sum(y_train)
scale_weight = sum_neg / sum_pos if sum_pos > 0 else 1

modelet = {
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "SVM (Linear)": SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=scale_weight, eval_metric='logloss', random_state=42),
    "Isolation Forest": IsolationForest(contamination=0.15, random_state=42)
}

rezultatet_benchmark = []
plt.figure(figsize=(9, 7))

for emer, model in modelet.items():
    start_train = time.time()
    
    if emer == "Isolation Forest":
        model.fit(X_train)
    else:
        model.fit(X_train, y_train)
    
    koha_train = time.time() - start_train
    start_test = time.time()
    
    if emer == "Isolation Forest":
        y_pred_if = model.predict(X_test)
        y_pred = [1 if x == -1 else 0 for x in y_pred_if]
    else:
        y_pred = model.predict(X_test)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{emer} (AUC = {roc_auc:.2f})')
        
    koha_test = (time.time() - start_test) * 1000 
    
    rezultatet_benchmark.append({
        "Modeli": emer,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "Train_Time (s)": round(koha_train, 4),
        "Inference_Time (ms)": round(koha_test, 2)
    })

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Klasifikues i Rastësishëm')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Shkalla e Pozitiveve te Rreme (FPR)')
plt.ylabel('Shkalla e Pozitiveve te Verteta (TPR)')
plt.title('Kurba ROC: Aftësia e Algoritmeve për të Dalluar Sulmet')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("1_ROC_Curve_Krahasimi.png", dpi=300)
plt.show()

df_benchmark = pd.DataFrame(rezultatet_benchmark)
print("\n=== TABELA 1: KRAHASIMI I ALGORITMEVE TË ZBULIMIT TË SULMEVE ===")
print(df_benchmark.to_string(index=False))

df_melted_metrics = df_benchmark.melt(id_vars="Modeli", value_vars=["Accuracy", "F1-Score"], var_name="Metrika", value_name="Vlera")
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted_metrics, x="Modeli", y="Vlera", hue="Metrika", palette="Set2")
plt.title("Krahasimi i Saktësisë (Accuracy) dhe F1-Score për secilin Model")
plt.ylabel("Rezultati (0 - 1)")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("2_Krahasimi_Metrikave_BarChart.png", dpi=300)
plt.show()

best_model = modelet["XGBoost"]
importances_sulme = best_model.feature_importances_

importances_sulme_shqip = pd.Series(importances_sulme, index=features_mekanizma).sort_values(ascending=False).rename(index=emrat_shqip)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances_sulme_shqip, y=importances_sulme_shqip.index, hue=importances_sulme_shqip.index, palette="Blues_r", legend=False)
plt.title("Rendesia e masave te sigurise ne parandalimin e sulmeve")
plt.xlabel("Rendesia (Sipas XGBoost)")
plt.ylabel("Mekanizmi i Sigurise")
plt.tight_layout()
plt.savefig("3_mekanizmat_vs_sulme.png", dpi=300)
plt.show()

y_koha = df['System_Downtime_Minutes']
reg = LinearRegression()
reg.fit(X_train, y_train)

coefs_koha = np.abs(reg.coef_)
importances_koha = coefs_koha / coefs_koha.sum()

importances_koha_shqip = pd.Series(importances_koha, index=features_mekanizma).sort_values(ascending=False).rename(index=emrat_shqip)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances_koha_shqip, y=importances_koha_shqip.index, hue=importances_koha_shqip.index, palette="Oranges_r", legend=False)
plt.title("Rendesia e masave ne uljen e kohes se nderprerjes")
plt.xlabel("Rendesia (Sipas Regresionit Linear)")
plt.ylabel("Mekanizmi i Sigurise")
plt.tight_layout()
plt.savefig("4_mekanizmat_vs_nderprerja.png", dpi=300)
plt.show()

df_avancuar = df[(df['AI_Anomaly_Detection'] == 1) & (df['Zero_Trust_Architecture'] == 1)]
df_bazik = df[(df['AI_Anomaly_Detection'] == 0) & (df['Zero_Trust_Architecture'] == 0)]

sulmet_krahasim = pd.DataFrame({'Sisteme Bazike': df_bazik[['DoS_Attacks_Per_Month', 'Malware_Attacks_Per_Month']].mean().values, 'Sisteme te Avancuara': df_avancuar[['DoS_Attacks_Per_Month', 'Malware_Attacks_Per_Month']].mean().values}, index=['Sulmet DoS', 'Sulmet Malware'])
koha_krahasim = pd.DataFrame({'Sisteme Bazike': [df_bazik['System_Downtime_Minutes'].mean()], 'Sisteme te Avancuara': [df_avancuar['System_Downtime_Minutes'].mean()]}, index=['Koha e Nderprerjes (min)'])

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
plt.savefig("5_testimi_hipotezes.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
kolonat_corr_eng = features_mekanizma + ['DoS_Attacks_Per_Month', 'DDoS_Attacks_Per_Month', 'Malware_Attacks_Per_Month', 'System_Downtime_Minutes']
kolonat_corr_shqip = [emrat_shqip[k] for k in kolonat_corr_eng]
corr_matrix = df[kolonat_corr_eng].corr()
corr_matrix.index = kolonat_corr_shqip
corr_matrix.columns = kolonat_corr_shqip

sns.heatmap(corr_matrix.loc[kolonat_corr_shqip[:5], kolonat_corr_shqip[5:]], 
            annot=True, cmap='RdYlGn', center=0, fmt=".2f", linewidths=.5)
plt.title("Matrica e Korrelacionit: Masat e Sigurisë vs Sulmet / Nderprerja")
plt.tight_layout()
plt.savefig("6_lidhja_korrelacioni.png", dpi=300)
plt.show()

try:
    df_pytesor = pd.read_excel("Pytesor_Rezultatet.xlsx")
    kolonat_pyq5 = df_pytesor.columns[5:10]
    df_pyq5_paster = df_pytesor[kolonat_pyq5].replace('*', np.nan).apply(pd.to_numeric, errors='coerce')
    mesataret_pytesor = df_pyq5_paster.mean()
    
    emrat_kolonave_pyq5 = {
        kolonat_pyq5[0]: 'Niveli i Enkriptimit',
        kolonat_pyq5[1]: 'Sistemi IDS (Zbulimi)',
        kolonat_pyq5[2]: 'Autentikimi Shumefaktor',
        kolonat_pyq5[3]: 'Zbulimi i Anomalive me AI',
        kolonat_pyq5[4]: 'Arkitektura Zero Trust'
    }
    mesataret_pytesor.index = mesataret_pytesor.index.map(emrat_kolonave_pyq5)
    
    tabela_krahasim = pd.DataFrame({
        'Mekanizmi': [emrat_shqip[f] for f in features_mekanizma],
        'Rendesia_Modeli': importances_sulme
    }).set_index('Mekanizmi')
    tabela_krahasim['Mesatarja_Pytesor'] = mesataret_pytesor
    
    scaler = MinMaxScaler()
    tabela_krahasim['Sintetike_Norm'] = scaler.fit_transform(tabela_krahasim[['Rendesia_Modeli']])
    tabela_krahasim['Pytesor_Norm'] = scaler.fit_transform(tabela_krahasim[['Mesatarja_Pytesor']])
    tabela_krahasim = tabela_krahasim.dropna(subset=['Sintetike_Norm', 'Pytesor_Norm'])
    
    df_melted = tabela_krahasim[['Sintetike_Norm', 'Pytesor_Norm']].reset_index().melt(id_vars='Mekanizmi', var_name='Burimi', value_name='Vlera')
    df_melted['Burimi'] = df_melted['Burimi'].map({'Sintetike_Norm': 'Inteligjenca Artificiale (XGBoost)', 'Pytesor_Norm': 'Perceptimi (Pyetësori)'})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, y='Mekanizmi', x='Vlera', hue='Burimi', palette=['#4C72B0', '#DD8452'])
    plt.title("Krahasimi i Inteligjencës Artificiale me Perceptimin e Studentëve")
    plt.xlabel("Rëndësia e Normalizuar (0 - 1)")
    plt.ylabel("Mekanizmi i Sigurise")
    plt.legend(title='Burimi i te dhenave')
    plt.tight_layout()
    plt.savefig("7_krahasimi_pytesor.png", dpi=300)
    plt.show()

except Exception as e:
    print(f"Grafiku i fundit u anashkalua: {e}")

df_benchmark.to_excel("Tabela_Krahasimi_Algoritmeve.xlsx", index=False)
print("\nTabela e krahasimit u ruajt me sukses si 'Tabela_Krahasimi_Algoritmeve.xlsx'")