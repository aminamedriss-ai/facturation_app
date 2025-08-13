import streamlit as st
import pandas as pd
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from io import BytesIO
import bcrypt
import json
import os
import hashlib
import base64
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import calendar
import math
import numpy as np
import io
from decimal import Decimal
from pathlib import Path
# 📷 Afficher un logo
st.set_page_config(
    page_title="Gestion de la Facturation",
    page_icon="logo2.png",  # chemin local ou URL
    layout="wide"
)
st.markdown("""
<style>
/* 🖼 Logo centré */
.logo-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 20px;
}

.logo {
    width: 200px;
    height: auto;
}

/* 🌈 Arrière-plan personnalisé + forcer mode sombre */
html, body, .stApp {
    background: #1d2e4e !important;
    font-family: 'Segoe UI', sans-serif;
    color-scheme: dark !important; /* Empêche l'inversion automatique */
    color: white !important;
}

/* 🖍️ Titre centré et coloré */
.main > div > div > div > div > h1 {
    text-align: center;
    color: #00796B !important;
}

/* 🧼 Nettoyage des bordures Streamlit */
.css-18e3th9 {
    padding: 1rem 0.5rem;
}

/* 🎨 Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1f3763 !important;
    color: white !important;
}

section[data-testid="stSidebar"] .css-1v3fvcr {
    color: white !important;
}

/* 🌈 Titres dans la sidebar */
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #e01b36 !important;
}

/* 🎨 Barre supérieure */
header[data-testid="stHeader"] {
    background-color: #06dbae !important;
    color: white !important;
}

/* 🧪 Supprimer la transparence */
header[data-testid="stHeader"]::before {
    content: "";
    background: none !important;
}

/* 📱 Correction mobile : forcer couleurs partout */
h1, h2, h3, p, span, label {
    color: white !important;
}

/* 🔵 Boutons bleu foncé forcés */
.stButton button {
    background-color: #2b2c36 !important; /* Bleu foncé */
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-weight: bold !important;
    -webkit-appearance: none !important; /* Évite style par défaut mobile */
    appearance: none !important;
}

.stButton button:hover {
    background-color: #43444e !important; /* Bleu plus clair au survol */
    color: white !important;
}
</style>
""", unsafe_allow_html=True)



# 🖼️ Ajouter un logo (remplacer "logo.png" par ton fichier ou une URL)
with open("logo.png", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

st.markdown(
    f"""
    <div class="logo-container">
        <img class="logo" src="data:image/png;base64,{encoded}">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "<h1 style='text-align: center;'>Bienvenue sur l'application de calcul des factures</h1>",
    unsafe_allow_html=True
)
USERS_FILE = "users.json"

# 🔒 Hachage du mot de passe
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 🔄 Charger les utilisateurs depuis le fichier JSON
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

# 💾 Sauvegarder les utilisateurs dans le fichier
def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# 📝 Interface d'inscription
def signup():
    st.subheader("Créer un compte")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    confirm = st.text_input("Confirmer le mot de passe", type="password")

    if st.button("Créer le compte"):
        if password != confirm:
            st.error("❌ Les mots de passe ne correspondent pas.")
            return

        users = load_users()
        if username in users:
            st.error("❌ Nom d'utilisateur déjà existant.")
        else:
            users[username] = hash_password(password)
            save_users(users)
            st.success("✅ Compte créé avec succès. Connectez-vous maintenant.")
            st.session_state["auth_mode"] = "login"
            st.rerun()

# 🔐 Interface de connexion
def login():
    st.subheader("Se connecter")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Connexion"):
        users = load_users()
        if username in users and users[username] == hash_password(password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success(f"Bienvenue {username} 👋")
            st.rerun()
        else:
            st.error("❌ Nom d'utilisateur ou mot de passe incorrect.")

# 🔓 Déconnexion
def logout():
    if st.sidebar.button("Se déconnecter"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.rerun()

# 🔄 Initialiser l'état
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["auth_mode"] = "login"

# 🔏 Si l'utilisateur n'est pas connecté, afficher l'interface d'authentification
if not st.session_state["authenticated"]:
    st.sidebar.title("🔐 Authentification")
    auth_mode = st.sidebar.radio("Choisissez une option :", ["Se connecter", "Créer un compte"])
    st.session_state["auth_mode"] = "login" if auth_mode == "Se connecter" else "signup"

    if st.session_state["auth_mode"] == "login":
        login()
    else:
        signup()

    st.stop()

# ✅ Si connecté, continuer l'application principale
# st.sidebar.success(f"Connecté en tant que **{st.session_state['username']}**")
logout()

# 🎉 Application principale ici
# st.title("Bienvenue sur l'application de calcule des factures")

def nettoyer_colonne(df, col):
    return (
        df[col]
        .astype(str)
        .str.replace("\u202f", "", regex=False)  # supprime espaces insécables
        .str.replace(" ", "", regex=False)       # supprime espaces normaux
        .str.replace(",", ".", regex=False)      # remplace virgule par point
        .str.replace(r"[^\d\.-]", "", regex=True)  # garde chiffres, . et -
        .replace("", "0")
        .astype(float)
    )
from math import floor
def calcul_base(row):
    salaire = Decimal(str(row["Salaire de base calcule"]))
    prime = Decimal(str(row["Prime mensuelle (Barème) (DZD)"]))
    panier = Decimal(str(row["Indemnité de panier (DZD)"]))
    transport  = Decimal(str(row["Indemnité de transport (DZD)"]))

    return floor((salaire + prime) - ((salaire + prime) * Decimal("0.09")) + (panier+transport))
def get_valeur(col_base, col_nouveau):
    if col_nouveau in df_client.columns:
        return nettoyer_colonne(df_client, col_nouveau).where(
            lambda x: x != 0, nettoyer_colonne(df_client, col_base)
        )
    else:
        return nettoyer_colonne(df_client, col_base)
def generer_facture_pdf(employe_dict, nom_fichier):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor("#0a5275"),
        spaceAfter=20,
    )
    header_style = ParagraphStyle(
        'Header',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=10,
    )

    # 🧾 En-tête : infos employé
    header_info = f"""
    <b>Nom:</b> {employe_dict.get("Nom", "")}<br/>
    <b>Prénom:</b> {employe_dict.get("Prénom", "")}<br/>
    <b>Année:</b> {employe_dict.get("Année", "")}<br/>
    <b>Titre du poste:</b> {employe_dict.get("Titre du poste", "")}<br/>
    <b>Durée CDD:</b> {employe_dict.get("Durée du CDD (Mois)", "")}<br/>
    <b>Établissement:</b> {employe_dict.get("Etablissement", "")}
    """
    elements.append(Paragraph("🧾 Facture individuelle de l'employé", title_style))
    elements.append(Paragraph(header_info, header_style))
    elements.append(Spacer(1, 12))

    # 📊 Lignes à afficher
    lignes = [
        "Salaire de base calcule", "Prime mensuelle calcule", "IFSP (20% du salaire de base)",
        "Prime exeptionnelle (10%) (DZD)", "Frais remboursement calcule",
        "Indemnité de panier calcule", "Indemnité de transport calcule", "Prime vestimentaire (DZD)",
        "Base cotisable","Base imposable au baréme", "IRG barème", "IRG 10%", "Salaire brut",
        "Retenue CNAS employé", "Salaire net", "CNAS employeur",
        "Cotisation œuvre sociale", "Taxe formation", "Taxe formation et os", "Masse salariale",
        "Coût congé payé", "Coût salaire", "Facture HT", "Facture TVA"
    ]

    mois = employe_dict.get("Mois", [])
    if isinstance(mois, str):
        mois = [mois]

    # 📊 Construction des données calculées
    tableau_data = [["Éléments"] + mois]
    for ligne in lignes:
        val = employe_dict.get(ligne, "")
        if isinstance(val, (int, float)):
            val = f"{val:,.2f}".replace(",", " ").replace(".", ",")  # Format DZD
        row = [ligne, val]
        tableau_data.append(row)

    # 🧱 Table PDF
    table = Table(tableau_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
    ]))

    elements.append(table)
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf




# 📌 Initialisation
if "clients" not in st.session_state:
    st.session_state.clients = []
if "selected_client" not in st.session_state:
    st.session_state.selected_client = None
if "full_df" not in st.session_state:
    st.session_state.full_df = None
if "data" not in st.session_state:
    st.session_state.data = {}


CLIENTS_FILE = Path("clients.json")
# st.title("👥 Gestion des clients et des employés")

# 📂 Charger la liste des clients
if CLIENTS_FILE.exists():
    with open(CLIENTS_FILE, "r", encoding="utf-8") as f:
        clients_list = json.load(f)
else:
    clients_list = [
       "Abbott", "Samsung", "Henkel", "G+D", "Maersk",
        "Cahors", "PMi", "Siemens", "Syngenta", "LG",
        "Epson", "EsteL", "JTI", "Siemens Energy", "Wilhelmsen",
        "Healthineers", "Contrat auto-entrepreneur", "Coca cola", "IPSEN", "SOGEREC","CCIS ex SOGEREC",
        "Roche", "Tango", "VARION"
    ]
    with open(CLIENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(clients_list, f, ensure_ascii=False, indent=2)
client_name = st.sidebar.selectbox(
    "Sélectionner un client",
    options=clients_list,
    index=None,  # <-- pas de sélection initiale
    placeholder="— Sélectionner un client —",
    key="client_select",
)
st.session_state.clients = clients_list

# 📁 Upload du fichier global
st.sidebar.subheader("📅 Charger le fichier récapitulatif (tous les clients)")
uploaded_csv = st.sidebar.file_uploader("Fichier CSV global", type=["csv"], key="csv_recap")

if uploaded_csv is not None:
    try:
        df_full = pd.read_csv(uploaded_csv, skiprows=2, decimal=",", thousands=" ") 
        st.write(df_full.head())
        st.session_state.full_df = df_full
        st.sidebar.success("✅ Fichier chargé avec succès !")
    except Exception as e:
        st.sidebar.error(f"❌ Erreur : {e}")

# ➕ Ajouter un nouveau client
st.sidebar.subheader("➕ Ajouter un nouveau client")
new_client = st.sidebar.text_input("Nom du nouveau client")
if st.sidebar.button("Ajouter"):
    if new_client and new_client not in st.session_state.clients:
        st.session_state.clients.append(new_client)
        with open(CLIENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.clients, f, ensure_ascii=False, indent=2)
        st.sidebar.success(f"Client '{new_client}' ajouté !")

# 🗑️ Supprimer un client avec confirmation
st.sidebar.subheader("🗑️ Supprimer un client")
client_to_delete = st.sidebar.selectbox("Choisir le client à supprimer", [""] + st.session_state.clients)

# Variable temporaire pour confirmation
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None

if st.sidebar.button("Supprimer"):
    if client_to_delete and client_to_delete in st.session_state.clients:
        st.session_state.confirm_delete = client_to_delete  # on garde le nom en mémoire

# Si un client est en attente de confirmation
if st.session_state.confirm_delete:
    st.warning(f"⚠️ Êtes-vous sûr de vouloir supprimer le client '{st.session_state.confirm_delete}' ?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Oui, supprimer"):
            st.session_state.clients.remove(st.session_state.confirm_delete)
            with open(CLIENTS_FILE, "w", encoding="utf-8") as f:
                json.dump(st.session_state.clients, f, ensure_ascii=False, indent=2)
            st.success(f"Client '{st.session_state.confirm_delete}' supprimé avec succès !")
            st.session_state.confirm_delete = None  # reset
    with col2:
        if st.button("❌ Annuler"):
            st.info("Suppression annulée.")
            st.session_state.confirm_delete = None


# 🧽 Sélection d'un client
# st.subheader("Sélectionnez un client")
# 🎯 Affichage des employés du client sélectionné
st.session_state.selected_client = client_name
if st.session_state.selected_client:
    st.markdown(f"## 👤 Données des employés pour **{st.session_state.selected_client.strip()}**")

    if st.session_state.full_df is not None:
        df = st.session_state.full_df.copy()
        df["Etablissement"] = df["Etablissement"].astype(str).str.strip()
        df_client = df[df["Etablissement"] == st.session_state.selected_client.strip()].copy()
        st.session_state.data[st.session_state.selected_client] = df_client.to_dict(orient="records")

        if not df_client.empty:
            mois_possibles = [mois.lower() for mois in calendar.month_name if mois]
            colonnes_mois = [col for col in df_client.columns if any(mois in col.lower() for mois in mois_possibles)]
            st.success(f"{len(df_client)} employés trouvés.")
            col1, col2, col3 = st.columns(3)
            with col1:
                fees_etablissement_pct = st.number_input("Fees etalent (%)", min_value=0.0, max_value=100.0, step=0.1, value=0.0)
                complementaire_sante_tarif = st.number_input("Complémentaire santé (DZD)", min_value=0.0, step=1.0, value=0.0)
                phone = st.number_input("Frais téléphone", min_value=0.0, step=1.0, value=0.0)

            with col2:
                indemnite_moto_tarif = st.number_input("Indemnité moto (DZD)", min_value=0.0, step=1.0, value=0.0)
                tap_tarif = st.number_input("TAP (DZD)", min_value=0.0, step=1.0, value=0.0)
                indemnite_zone = st.number_input("Indemnité de zone (DZD)", min_value=0.0, step=1.0, value=0.0)
            with col3:
                tva_tarif = st.number_input("TVA (%)", min_value=0.0, max_value=100.0, step=0.1, value=0.0)
                frais_divers_tarif = st.number_input("Frais divers + transport (Yassir) (DZD)", min_value=0.0, step=1.0, value=0.0)
                augmentation = st.number_input("Augmentation (%)", min_value=0.0, max_value=100.0, step=0.1, value=0.0)
                jours_mois = st.number_input("Jours mois", min_value=0.0, step=1.0, value=0.0)

            # 2. Nettoyage des colonnes
            cols_to_float = [
                "Salaire de base (DZD)", "Prime mensuelle (Barème) (DZD)", "IFSP (20% du salaire de base)",
                "Prime exeptionnelle (10%) (DZD)", "Frais de remboursement (Véhicule) (DZD)", 
                "Indemnité de panier (DZD)", "Indemnité de transport (DZD)", "Nouveau Salaire de base (DZD)",
                "Prime vestimentaire (DZD)", "Nouvelle Indemnité de panier (DZD)",  "Nouvelle Indemnité de transport (DZD)",
                 "Nouvelle Prime mensuelle (DZD)", "Nouveaux Frais de remboursement (Véhicule) (DZD)","Prime vestimentaire (DZD)", " Indémnité Véhicule (DZD)",
                 "Absence (Jour)","Absence Maladie (Jour)","Absence Maternité (Jour)", "Absence Mise à pied (Jour)", "Jours de congé (Jour)"
            ]
            for col in cols_to_float:
                if col in df_client.columns:
                    df_client[col] = nettoyer_colonne(df_client, col)
                else:
                    df_client[col] = 0.0
            absences_total = (
                df_client["Absence (Jour)"] +
                df_client["Absence Maladie (Jour)"] +
                df_client["Absence Maternité (Jour)"] +
                df_client["Absence Mise à pied (Jour)"] +
                df_client["Jours de congé (Jour)"]
            )
            # 3. Calculs (une seule fois)
            df_client["Salaire de base calcule"] = (get_valeur("Salaire de base (DZD)", "Nouveau Salaire de base (DZD)")+df_client["IFSP (20% du salaire de base)"])
            df_client["Indemnité de panier calcule"] = get_valeur("Indemnité de panier (DZD)", "Nouvelle Indemnité de panier (DZD)")
            df_client["Indemnité de transport calcule"] = get_valeur("Indemnité de transport (DZD)", "Nouvelle Indemnité de transport (DZD)")
            df_client["Prime mensuelle calcule"] = get_valeur("Prime mensuelle (DZD)", "Nouvelle Prime mensuelle (DZD)")
            df_client["Frais remboursement calcule"] = get_valeur("Frais de remboursement (Véhicule) (DZD)", "Nouveaux Frais de remboursement (Véhicule) (DZD)")
            # df_client["Salaire de base calcule"] = (
            #     df_client["Salaire de base calcule"] * (1 + (augmentation / 100))
            # ) * ((jours_mois - absences_total) / jours_mois)
            # df_client["Indemnité de panier calcule"]= df_client["Indemnité de panier calcule"]- ((df_client["Indemnité de panier calcule"]/22)*absences_total)+(df_client["Indemnité de panier calcule"]/22)
            # df_client["Indemnité de transport calcule"]= df_client["Indemnité de transport calcule"] -((df_client["Indemnité de transport calcule"]/22)*absences_total)+(df_client["Indemnité de transport calcule"]/22)
            # df_client["Prime vestimentaire (DZD)"]=df_client["Prime vestimentaire (DZD)"]-((df_client["Prime vestimentaire (DZD)"]/22)*absences_total)+(df_client["Prime vestimentaire (DZD)"]/22)
            df_client["Base cotisable"] = (
                df_client["Prime exeptionnelle (10%) (DZD)"] + indemnite_zone + 
                df_client["Prime mensuelle calcule"] + df_client["Salaire de base calcule"]
            )
            df_client["Base imposable 10%"] = df_client["Prime exeptionnelle (10%) (DZD)"] * 0.91
            df_client["Retenue CNAS employé"] = df_client["Base cotisable"] * 0.09
            if df_client["Etablissement"].iloc[0] == "Henkel":    
                df_client["Base imposable au baréme"]  = ((((df_client["Salaire de base calcule"]+ df_client["Prime mensuelle calcule"])-((df_client["Salaire de base calcule"]+ df_client["Prime mensuelle calcule"])*0.09))+df_client["Indemnité de panier calcule"])/10)*10
                
            else:
                df_client["Base imposable au baréme"] = (np.floor(((df_client["Base cotisable"] - df_client["Prime exeptionnelle (10%) (DZD)"]- indemnite_zone) * 0.91+ (df_client["Indemnité de panier calcule"]) + df_client["Indemnité de transport calcule"] + df_client["Prime vestimentaire (DZD)"]+df_client[ " Indémnité Véhicule (DZD)"])/ 10) * 10)
            def irg_bareme(base):
                b = np.ceil(base / 10) * 10  # PLAFOND(...;10) en Excel
                
                if b < 30000:
                    return 0
                elif 30000 <= b <= 30860:
                    return (((( (b - 20000) * 0.23) - 1000) * 137/51) - (27925/8))
                elif 30860 < b < 35000:
                    return (((( (b - 20000) * 0.23) * 0.60) * 137/51) - (27925/8))
                elif 35000 <= b < 36310:
                    return (b - 20000) * 0.23 * 0.60
                elif 36310 <= b < 40000:
                    return (b - 20000) * 0.23 - 1500
                elif 40000 <= b < 80000:
                    return (b - 40000) * 0.27 + 3100
                elif 80000 <= b < 160000:
                    return (b - 80000) * 0.30 + 13900
                elif 160000 <= b < 320000:
                    return (b - 160000) * 0.33 + 37900
                else:  # b >= 320000
                    return (b - 320000) * 0.35 + 90700

            df_client["IRG barème"] = df_client["Base imposable au baréme"].apply(irg_bareme)
            df_client["IRG 10%"] = df_client["Base imposable 10%"] * 0.10
            df_client["Salaire brut"] = (
                df_client["Base cotisable"] +
                df_client["Indemnité de panier calcule"] + df_client["Indemnité de transport calcule"] +df_client["Prime vestimentaire (DZD)"]+
                df_client["Frais remboursement calcule"] + df_client[ " Indémnité Véhicule (DZD)"]
            )
            df_client["Salaire net"] = (
                df_client["Salaire brut"] -
                df_client["Retenue CNAS employé"] -
                df_client["IRG barème"] -
                df_client["IRG 10%"]
            )
            df_client["CNAS employeur"] = df_client["Base cotisable"] * 0.26
            if  df_client["Etablissement"].iloc[0] == "Henkel":
                df_client["Taxe formation et os"] = (df_client["Salaire de base calcule"] + df_client["Prime mensuelle calcule"]+ df_client["Indemnité de panier calcule"] + df_client["Indemnité de transport calcule"] +df_client["Prime vestimentaire (DZD)"]) * 0.04
                df_client["Cotisation œuvre sociale"] = 0
                df_client["Taxe formation"] = 0
            elif df_client["Etablissement"].iloc[0] == "Maersk" :
                df_client["Taxe formation et os"] = (df_client["Base cotisable"]) * 0.03
                df_client["Cotisation œuvre sociale"] = 0
                df_client["Taxe formation"] = 0
            else:
                df_client["Cotisation œuvre sociale"] = df_client["Salaire brut"] * 0.02
                df_client["Taxe formation"] = df_client["Salaire brut"] * 0.02
                df_client["Taxe formation et os"] =0
            df_client["Masse salariale"] = (
                df_client["Salaire brut"] +
                df_client["CNAS employeur"] +
                df_client["Cotisation œuvre sociale"] +
                df_client["Taxe formation"]
            )
            
            if df_client["Etablissement"].iloc[0] == "Henkel":
                df_client["Coût salaire"] = (
                    df_client["Salaire net"]
                    + df_client["Taxe formation et os"]
                    + df_client["CNAS employeur"]
                    + df_client["IRG 10%"]
                    + df_client["IRG barème"]
                    + df_client["Retenue CNAS employé"]
                    + phone
                )
                df_client["Coût congé payé"] = df_client["Coût salaire"] / 30 * 2.5
                fees_multiplicateur = 1 + (fees_etablissement_pct / 100)
                df_client["Facture HT"] = ((df_client["Coût salaire"] + df_client["Coût congé payé"]+ tap_tarif)* fees_multiplicateur) 
            elif df_client["Etablissement"].iloc[0] == "LG":
                df_client["Coût salaire"] = (
                    df_client["Salaire net"]
                    + df_client["Cotisation œuvre sociale"]
                    + df_client["Taxe formation"]
                    + df_client["CNAS employeur"]
                    + df_client["IRG 10%"]
                    + df_client["IRG barème"]
                    + df_client["Retenue CNAS employé"]
                    + phone
                )
                fees_multiplicateur = 1 + (fees_etablissement_pct / 100)
                df_client["Facture HT"] = (df_client["Coût salaire"] * fees_multiplicateur) + tap_tarif
            elif df_client["Etablissement"].iloc[0] == "Maersk":
                df_client["Coût salaire"] = (
                    df_client["Salaire net"]
                    + df_client["Taxe formation et os"]
                    + df_client["CNAS employeur"]
                    + df_client["IRG 10%"]
                    + df_client["IRG barème"]
                    + df_client["Retenue CNAS employé"]
                )

                df_client["Coût congé payé"] = df_client["Coût salaire"] / 30 * 2.5
                fees_multiplicateur = 1 + (fees_etablissement_pct / 100)
                df_client["Facture HT"] = ((df_client["Coût salaire"] + df_client["Coût congé payé"]+ tap_tarif)* fees_multiplicateur) 
            elif df_client["Etablissement"].iloc[0] == "G+D":
                df_client["Coût salaire"] = df_client["Salaire de base calcule"] + df_client["Indemnité de panier calcule"] + df_client["Indemnité de transport calcule"] +df_client["Prime vestimentaire (DZD)"]+df_client["Frais remboursement calcule"]+df_client["Prime exeptionnelle (10%) (DZD)"]
                fees_multiplicateur = 1 + (fees_etablissement_pct / 100)
                df_client["Facture HT"] = (df_client["Coût salaire"] * fees_multiplicateur) + tap_tarif
            else:
                df_client["Coût congé payé"] = df_client["Masse salariale"] * (2.5 / 30)
                df_client["Coût salaire"] = (
                    df_client["Masse salariale"]
                    + df_client["Coût congé payé"]
                    + complementaire_sante_tarif
                    + frais_divers_tarif
                    + phone
                )
                fees_multiplicateur = 1 + (fees_etablissement_pct / 100)
                df_client["Facture HT"] = (df_client["Coût salaire"] * fees_multiplicateur) + tap_tarif

           


            # fees_multiplicateur = 1 + (fees_etablissement_pct / 100)
            # df_client["Facture HT"] = (df_client["Coût salaire"] * fees_multiplicateur) + tap_tarif
            tva_multiplicateur = 1+ (tva_tarif/100)
            df_client["Facture TVA"] = df_client["Facture HT"] * tva_multiplicateur
           
            st.write(df_client.head())

# Construire employe_data APRÈS les calculs
            employe_data = []
            mois_possibles = [mois.lower() for mois in calendar.month_name if mois]
            colonnes_mois = [col for col in df_client.columns if any(m in col.lower() for m in mois_possibles)]

            for _, row in df_client.iterrows():
                data_dict = row.to_dict()
                data_dict["Mois"] = colonnes_mois

                # Prendre les valeurs calculées de df_client pour chaque ligne
                data_dict["data"] = {
                    ligne: [row.get(ligne, "") for _ in colonnes_mois]
                    for ligne in [
                        "Salaire de base calcule", "Prime mensuelle calcule", "IFSP (20% du salaire de base)",
                        "Prime exeptionnelle (10%) (DZD)", "Frais remboursement calcule",
                        "Indemnité de panier calcule", "Indemnité de transport calcule", "Prime vestimentaire (DZD)",
                        "Base cotisable","Base imposable au baréme", "IRG barème", "IRG 10%", "Salaire brut",
                        "Retenue CNAS employé", "Salaire net", "CNAS employeur",
                        "Cotisation œuvre sociale", "Taxe formation","Taxe formation et os", "Masse salariale",
                        "Coût congé payé", "Coût salaire", "Facture HT","Facture TVA"
                    ]
                }
                employe_data.append(data_dict)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_client.to_excel(writer, index=False, sheet_name='Calculs')
                
                workbook = writer.book
                worksheet = writer.sheets['Calculs']

                # 🎨 Style d'entête
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'middle',
                    'align': 'center',
                    'fg_color': '#0a5275',
                    'font_color': 'white',
                    'border': 1
                })

                # Appliquer style entête
                for col_num, value in enumerate(df_client.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                # 📌 Ajuster largeur colonnes
                for i, col in enumerate(df_client.columns):
                    col_width = max(df_client[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, col_width)

                # 💰 Format monétaire pour colonnes DZD
                money_format = workbook.add_format({'num_format': '#,##0.00 DZD', 'border': 1})
                for i, col in enumerate(df_client.columns):
                    if "DZD" in col or "Salaire" in col or "Coût" in col or "Facture" in col:
                        worksheet.set_column(i, i, None)

            # 📥 Bouton de téléchargement Excel
            st.download_button(
                label="📊 Télécharger les résultats en Excel",
                data=output.getvalue(),
                file_name=f"{st.session_state.selected_client}_calculs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.markdown("### 📥 Télécharger la facture PDF par employé")
            for idx, row in df_client.iterrows():
                nom = str(row.get("Nom", f"employe_{idx}")).strip().replace(" ", "_")
                matricule = str(row.get("Matricule", f"id_{idx}")).strip()
                pdf = generer_facture_pdf(row.to_dict(), f"{matricule}_{nom}_facture.pdf")
                st.download_button(
                    label=f"📄 {nom}",
                    data=pdf,
                    file_name=f"{nom}_facture.pdf",
                    mime="application/pdf",
                    key=f"pdf_{matricule}_{idx}"
                )

        else:
            st.warning("⚠️ Aucun employé trouvé pour ce client ")
          
    else:
        st.info("Veuillez d'abord téléverser le fichier récapitulatif global dans la barre latérale.")
