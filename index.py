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
from rapidfuzz import process, fuzz
from reportlab.platypus import Image
from supabase import create_client, Client
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
def trouver_client(client_name, df):
    """Retourne le dataframe filtré pour le client choisi."""
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df["Etablissement"] = df["Etablissement"].astype(str).str.strip()
    return df[df["Etablissement"].str.lower() == client_name.strip().lower()].copy()
from reportlab.platypus import Image, Table, TableStyle, Spacer, Paragraph
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
def _make_image_flowable(path, target_height):
    """Retourne un Image flowable redimensionné en conservant le ratio.
       Si le fichier n'existe pas, retourne un Spacer de la même hauteur."""
    if not path or not os.path.exists(path):
        return Spacer(1, target_height)
    try:
        img = ImageReader(path)
        iw, ih = img.getSize()
        ratio = target_height / ih
        width = iw * ratio
        return Image(path, width=width, height=target_height)
    except Exception as e:
        print(f"⚠ Impossible de charger l'image {path}: {e}")
        return Spacer(1, target_height)
# def calcul_jours_ouvres(row):
#     # Ignorer si une des deux dates est NaN
#     if pd.isna(row["Date départ congé"]) or pd.isna(row["Date de reprise"]):
#         return 0

#     # Conversion en datetime
#     date_debut = pd.to_datetime(row["Date départ congé"], dayfirst=True, errors="coerce")
#     date_fin = pd.to_datetime(row["Date de reprise"], dayfirst=True, errors="coerce")

#     if pd.isna(date_debut) or pd.isna(date_fin):
#         return 0

#     # DEBUG: Afficher les dates
#     print(f"Date début: {date_debut}, Date fin: {date_fin}")
    
#     # Générer les dates entre début et fin (exclure la date de reprise)
#     toutes_les_dates = pd.date_range(start=date_debut, end=date_fin - pd.Timedelta(days=1), freq="D")
    
#     # DEBUG: Afficher toutes les dates générées
#     print(f"Toutes les dates générées: {list(toutes_les_dates)}")
#     print(f"Nombre total de dates: {len(toutes_les_dates)}")

#     # Exclure vendredi (4) et samedi (5) -> week-end en Algérie
#     jours_ouvres = toutes_les_dates[~toutes_les_dates.weekday.isin([4, 5])]
    
#     # DEBUG: Afficher les jours ouvrés
#     print(f"Jours ouvrés: {list(jours_ouvres)}")
#     print(f"Nombre jours ouvrés: {len(jours_ouvres)}")
#     print("---")

#     return len(jours_ouvres)


def calcul_joursstc_ouvres(row):
    val = row["Date du dernier jour travaillé"]
    nbr_jours_stc = row["Nbr jours STC (jours)"]
    
    date_debut = pd.to_datetime(val, dayfirst=True, errors="coerce")
    if pd.isna(date_debut) or nbr_jours_stc == 0:
        return 0

    start = (date_debut + pd.Timedelta(days=1)).normalize()
    end = start + pd.Timedelta(days=nbr_jours_stc - 1)

    # Générer toutes les dates de la période
    toutes_les_dates = pd.date_range(start=start, end=end, freq="D")
    
    # Filtrer les jours ouvrés (exclure vendredi=4 et samedi=5)
    jours_ouvres = sum(1 for date in toutes_les_dates if date.weekday() not in [4, 5])
    
    return jours_ouvres

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import os
from openpyxl.drawing.image import Image as XLImage
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter

def generer_facture_excel(employe_dict, nom_fichier, logos_folder="facturation_app/Logos"):
    # Créer un nouveau classeur Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "Facturation"
    
    # 📌 Styles de base
    header_font = Font(bold=True, size=14, color="000000")
    normal_font_black = Font(size=11, color="000000")
    normal_font_white = Font(size=11, color="FFFFFF")
    header_fill = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
    data_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    border = Border(left=Side(style='thin'), right=Side(style='thin'),
                    top=Side(style='thin'), bottom=Side(style='thin'))
    center_alignment = Alignment(horizontal='center', vertical='center')
    left_alignment = Alignment(horizontal='left', vertical='center')
    COL_OFFSET = 4
    # 📌 Mapping des couleurs personnalisées
    color_map = {
        # Bleu clair
        "Base cotisable": "9fc5e8",
        "Retenue CNAS employé": "9fc5e8",
        "Base imposable au baréme": "9fc5e8",
        "IRG barème": "9fc5e8",
        "Base imposable 10%": "9fc5e8",
        "IRG 10%": "9fc5e8",
        "Salaire brut": "9fc5e8",
        "CNAS employeur": "9fc5e8",
        "Cotisation œuvre sociale": "9fc5e8",
        "Taxe formation": "9fc5e8",
        "Taxe formation et os": "9fc5e8",
        "Frais téléphone": "9fc5e8",
        "Frais de transport (Yassir)": "9fc5e8",
        "Frais divers": "9fc5e8",
        "Coût congé payé": "9fc5e8",
        "Taux complément santé (DZD)": "9fc5e8",
        "Fees etalent": "9fc5e8",
        "TAP": "9fc5e8",

        # Bleu foncé
        "Salaire net": "25488e",
        "Masse salariale": "25488e",
        "Coût salaire": "25488e",

        # Rouge
        "Facture HT": "e11b36",

        # Vert foncé
        "NDF": "284052",
        "Facture TVA": "284052",
        "Facture TTC": "284052",
    }

    # 📌 Lignes qui doivent avoir le texte en blanc (fonds foncés uniquement)
    white_text_lines = {
        "Salaire net", "Masse salariale", "Coût salaire",
        "Facture HT", "NDF", "Facture TVA", "Facture TTC"
    }
    
    # 📌 Logo du client
    etablissement = str(employe_dict.get("Etablissement", "")).strip()
    logo_path = os.path.join(logos_folder, f"{etablissement}.png")

    if os.path.exists(logo_path):
        try:
            logo = XLImage(logo_path)
            logo.width = 400   # largeur ajustable
            logo.height = 130   # hauteur ajustable
            ws.add_image(logo, f"{get_column_letter(COL_OFFSET+4)}1")  
        except Exception as e:
            print(f"⚠️ Impossible d’insérer le logo pour {etablissement}: {e}")
    else:
        print(f"⚠️ Logo introuvable pour {etablissement} ({logo_path})")
    ws.merge_cells(start_row=1, start_column=COL_OFFSET+1, end_row=1, end_column=COL_OFFSET+6)
    # ws.cell(row=1, column=COL_OFFSET+1, value="FICHE DE FACTURATION").font = Font(bold=True, size=16, color="000000")
    ws.cell(row=1, column=COL_OFFSET+1).alignment = center_alignment
    infos_employe = [
        ["Nom:", employe_dict.get("Nom", "")],
        ["Prénom:", employe_dict.get("Prénom", "")],
        ["Année:", employe_dict.get("Année", "")],
        ["Titre du poste:", employe_dict.get("Titre du poste", "")],
        ["Durée CDD:", employe_dict.get("Durée du CDD (Mois)", "")],
        ["Établissement:", employe_dict.get("Etablissement", "")]
    ]
    
    for i, (label, value) in enumerate(infos_employe, start=3):
        ws.cell(row=i, column=COL_OFFSET+0, value=label).font = Font(bold=True)
        ws.cell(row=i, column=COL_OFFSET+1, value=value).font = normal_font_black
    
    # 📌 Détermination des lignes selon l’établissement
    etablissement = str(employe_dict.get("Etablissement", "")).strip()
    clients_simples = ["Abbott", "Samsung"]
    client_sante = ["Siemens", "Healthineers","Siemens Energy", "Siemens Healthineers Oncology" ,"Tango","Roche","CCIS ex SOGEREC","JTI","Philip Morris International","Wilhelmsen", "IPSEN", "LG"]
    client_os = ["Maersk", "Henkel"]
    client_change = ["Epson"]
    client_change_phone = ["Cahors"]
    client_ndf = ["Syngenta"]
    client_gd = ["G+D"]
    
    if etablissement in clients_simples:
        lignes = [
           "Salaire de base", "Prime mensuelle", "Prime exeptionnelle (10%) (DZD)", "Indemnités Non Cotisable - Mensuelle | Panier, Transport", 
           "Frais remboursement","Base cotisable", "Retenue CNAS employé", "Base imposable au baréme","IRG barème","Base imposable 10%", "IRG 10%",
            "Salaire net","Salaire brut", "CNAS employeur", "Cotisation œuvre sociale", "Taxe formation", "Masse salariale", 
            "Coût congé payé","Coût salaire","Fees etalent", "Facture HT","Facture TVA", "Facture TTC"
        ]
    elif etablissement in client_sante:
        lignes = [
           "Salaire de base", "Prime mensuelle", "Prime exeptionnelle (10%) (DZD)","Prime vestimentaire (DZD)", "Indemnités Non Cotisable - Mensuelle | Panier, Transport", 
           "Frais remboursement","Base cotisable", "Retenue CNAS employé", "Base imposable au baréme","IRG barème","Base imposable 10%", "IRG 10%",
            "Salaire net","Salaire brut", "CNAS employeur", "Cotisation œuvre sociale", "Taxe formation", "Masse salariale", 
            "Coût congé payé","Taux complément santé (DZD)","Coût salaire","Fees etalent", "Facture HT","Facture TVA", "Facture TTC"
        ]
    elif etablissement in client_os:
        lignes = [
           "Salaire de base", "Prime mensuelle", "Prime exeptionnelle (10%) (DZD)", "Indemnités Non Cotisable - Mensuelle | Panier, Transport", 
           "Frais remboursement","Base cotisable", "Retenue CNAS employé", "Base imposable au baréme","IRG barème","Base imposable 10%", "IRG 10%",
            "Salaire net","Salaire brut", "CNAS employeur","Taxe formation et os", 
            "Coût congé payé","Taux complément santé (DZD)","Coût salaire","Fees etalent","TAP", "Facture HT","Facture TVA", "Facture TTC"
        ]
    elif etablissement in client_change:
        lignes = [
           "Salaire de base", "Prime mensuelle", "Prime exeptionnelle (10%) (DZD)", "Indemnités Non Cotisable - Mensuelle | Panier, Transport", 
           "Frais remboursement","Base cotisable", "Retenue CNAS employé", "Base imposable au baréme","IRG barème","Base imposable 10%", "IRG 10%",
            "Salaire net","Salaire brut", "CNAS employeur", "Cotisation œuvre sociale", "Taxe formation", "Masse salariale", 
            "Coût congé payé","Frais téléphone",
             "Frais de transport (Yassir)","Frais divers","Coût salaire","Fees etalent", "Facture HT","Facture TVA", "Facture TTC"
        ]
    elif etablissement in client_change_phone:
        lignes = [
           "Salaire de base", "Prime mensuelle", "Prime exeptionnelle (10%) (DZD)", "Indemnités Non Cotisable - Mensuelle | Panier, Transport", 
           "Frais remboursement","Base cotisable", "Retenue CNAS employé", "Base imposable au baréme","IRG barème","Base imposable 10%", "IRG 10%",
            "Salaire net","Salaire brut", "CNAS employeur", "Cotisation œuvre sociale", "Taxe formation", "Masse salariale", 
            "Coût congé payé","Coût salaire","Fees etalent", "Facture HT","Facture TVA", "Facture TTC"
        ]
    elif etablissement in client_ndf:
         lignes = [
           "Salaire de base", "Prime mensuelle", "Prime exeptionnelle (10%) (DZD)", "Prime vestimentaire (DZD)","Indemnités Non Cotisable - Mensuelle | Panier, Transport", 
           "Frais remboursement","Base cotisable", "Retenue CNAS employé", "Base imposable au baréme","IRG barème","Base imposable 10%", "IRG 10%",
            "Salaire net","Salaire brut", "CNAS employeur", "Cotisation œuvre sociale", "Taxe formation", "Masse salariale", 
            "Coût congé payé","Coût salaire","Fees etalent", "Facture HT","NDF","Facture TVA", "Facture TTC"
        ]
    elif etablissement in client_gd:
        lignes = [
            "Salaire de base", "Prime mensuelle", "Prime exeptionnelle (10%) (DZD)", "Indemnités Non Cotisable - Mensuelle | Panier, Transport", 
            "Salaire net", "Facture HT","Facture TVA", "Facture TTC"
        ]
    else:
        lignes = [ "Salaire de base","IFSP (20% du salaire de base)", "Prime mensuelle",  "Prime exeptionnelle (10%) (DZD)", "Indemnité de panier",
                "indémnité Véhicule",  "Indemnités Non Cotisable - Mensuelle | Panier, Transport", 
                "Frais remboursement","Base cotisable", "Retenue CNAS employé", "Base imposable au baréme","IRG barème","Base imposable 10%", "IRG 10%",
                "Salaire net","Salaire brut", "CNAS employeur", "Cotisation œuvre sociale", "Taxe formation","Taxe formation et os", "Masse salariale", 
                "Coût congé payé","Taux complément santé (DZD)","Coût salaire","Fees etalent", "Facture HT", "Facture TTC" ]
    
    # 📌 RÉCUPÉRER LES DONNÉES POUR LES MOIS
    mois_data = {}
    for key, value in employe_dict.items():
        if '_' in key:
            ligne_nom, mois_nom = key.rsplit('_', 1)
            mois_data.setdefault(mois_nom, {})[ligne_nom] = value
    mois_disponibles = list(mois_data.keys()) or ["Août", "Septembre"]
    
    # 📌 Création du tableau
    start_row = 10
    ws.cell(row=start_row, column=COL_OFFSET+0, value="Éléments")
    
    for col, mois in enumerate(mois_disponibles, start=1):
        ws.cell(row=start_row, column=COL_OFFSET+col, value=mois)
    
    for col in range(len(mois_disponibles) + 1):
        cell = ws.cell(row=start_row, column=COL_OFFSET+col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_alignment
        cell.border = border
    
    # 📌 Données ligne par ligne
    for row, ligne in enumerate(lignes, start=1):
        current_row = start_row + row
        
        # Vérifier si cette ligne a une couleur spéciale
        fill_color = color_map.get(ligne)
        fill_style = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid") if fill_color else None
        font_color = "FFFFFF" if ligne in white_text_lines else "000000"
        
        # 📌 Cellule titre (colonne A)
        cell_titre = ws.cell(row=current_row, column=COL_OFFSET+0, value=ligne)
        cell_titre.font = Font(bold=True, color=font_color)
        cell_titre.alignment = left_alignment
        cell_titre.border = border
        if fill_style:
            cell_titre.fill = fill_style
        
        # 📌 Valeurs par mois
        for col, mois in enumerate(mois_disponibles, start=1):
            val = mois_data.get(mois, {}).get(ligne, "N/A")
            if isinstance(val, (int, float)):
                val = f"{val:,.2f}".replace(",", " ").replace(".", ",")
            
            cell = ws.cell(row=current_row, column=COL_OFFSET+col, value=val)
            cell.font = Font(size=11, color=font_color)
            cell.alignment = center_alignment
            cell.border = border
            
            if fill_style:
                cell.fill = fill_style
            elif row % 2 == 0:  # alternance gris clair
                cell.fill = data_fill

    
    # 📌 Largeur colonnes
    
    for col in range(COL_OFFSET, COL_OFFSET + len(mois_disponibles) + 2):
        ws.column_dimensions[get_column_letter(col)].width = 40
    
    # 📌 Sauvegarde
    if not nom_fichier.endswith('.xlsx'):
        nom_fichier += '.xlsx'
    wb.save(nom_fichier)
    return nom_fichier





USERS_FILE = "users.json"
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
# 🔒 Hachage du mot de passe
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 📝 Inscription
def signup(email, password):
    response = supabase.auth.sign_up({"email": email, "password": password})
    return response

# 🔐 Connexion
def login(email, password):
    response = supabase.auth.sign_in_with_password({"email": email, "password": password})
    return response

# 🚪 Déconnexion
def logout():
    supabase.auth.sign_out()
    st.session_state["authenticated"] = False
    st.session_state["user"] = None

# --- Interface ---
# st.title("🔐 Auth avec Supabase")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    mode = st.radio("Choisissez :", ["Connexion", "Créer un compte"])

    email = st.text_input("Email")
    password = st.text_input("Mot de passe", type="password")

    if mode == "Créer un compte":
        if st.button("Créer un compte"):
            res = signup(email, password)
            if res.user:
                st.success("✅ Compte créé ! Vérifie ton email.")
            else:
                st.error(f"Erreur : {res}")
    else:
        if st.button("Connexion"):
            res = login(email, password)
            if res.user:
                st.session_state["authenticated"] = True
                st.session_state["user"] = res.user
                st.rerun()
            else:
                st.error("❌ Email ou mot de passe incorrect.")

else:
    with st.sidebar:
        st.success(f"Bienvenue {st.session_state['user'].email} 👋")
        if st.button("Se déconnecter"):
            logout()
            st.rerun()

# 🎉 Application principale ici
# st.title("Bienvenue sur l'application de calcule des factures")



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
            "Cahors", "Philip Morris International", "Siemens", "Syngenta", "LG",
            "Epson", "EsteL", "JTI", "Siemens Energy", "Wilhelmsen",
            "Healthineers", "Contrat auto-entrepreneur", "Coca Cola", "IPSEN", "SOGEREC","CCIS ex SOGEREC",
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
    st.session_state.selected_client = client_name
    # 📁 Upload du fichier global
    GLOBAL_CSV = "data_global.csv"
    st.sidebar.subheader("📅 Charger le fichier récapitulatif (tous les clients)")
    uploaded_csv = st.sidebar.file_uploader("Fichier CSV global", type=["csv"], key="csv_recap")

    if uploaded_csv is not None:
        try:
            df_full = pd.read_csv(uploaded_csv, skiprows=2, decimal=",", thousands=" ") 
            st.write(df_full.head())
            st.session_state.full_df = df_full
            st.sidebar.success("✅ Fichier chargé avec succès !")

            # Extraire le mois
            if "Mois" not in df_full.columns:
                mois = uploaded_csv.name.split("_")[1].replace(".csv", "")  
                df_full["Mois"] = mois
            else:
                mois = df_full["Mois"].iloc[0]

            # Vérifier si le global existe
            if os.path.exists(GLOBAL_CSV):
                df_global = pd.read_csv(GLOBAL_CSV, sep=";")

                # ⚡ Supprimer les anciennes lignes du même mois
                df_global = df_global[df_global["Mois"] != mois]

                # Ajouter les nouvelles lignes
                df_concat = pd.concat([df_global, df_full], ignore_index=True)
            else:
                df_concat = df_full

            # ✅ Sauvegarde (toujours exécutée)
            df_concat.to_csv(GLOBAL_CSV, sep=";", index=False, encoding="utf-8-sig")
            st.success(f"✅ Données mises à jour pour {mois} dans {GLOBAL_CSV}")

            # ✅ Affichage (toujours exécuté)
            # st.dataframe(df_concat)

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


    if st.session_state.selected_client:
        st.markdown(f"## 👤 Données des employés pour **{st.session_state.selected_client.strip()}**")

        if st.session_state.full_df is not None:
            # df = st.session_state.full_df.copy()
            df = pd.read_csv(GLOBAL_CSV, sep=";")  # toujours le fichier global
            df["Etablissement"] = df["Etablissement"].astype(str).str.strip()
            df_client = trouver_client(st.session_state.selected_client, df)


            st.session_state.data[st.session_state.selected_client] = df_client.to_dict(orient="records")

            if not df_client.empty:
                # ------------------------------------------------
                # Partie calculs et préparation des données
                # ------------------------------------------------
                mois_possibles = [mois.lower() for mois in calendar.month_name if mois]
                colonnes_mois = [col for col in df_client.columns if any(mois in col.lower() for mois in mois_possibles)]
                nb_employes = df_client["N°"].nunique()
                st.success(f"{nb_employes} employés trouvés.")
                col1, = st.columns(1) 
                with col1:
                    jours_mois = st.number_input("Jours mois", min_value=28.0, max_value=31.0, step=1.0, value=30.0)

                Observation = st.text_input("Observation", value="")

                # 2. Nettoyage des colonnes
                cols_to_float = [
                    "Salaire de base (DZD)", "Prime mensuelle (Barème) (DZD)", "IFSP (20% du salaire de base)",
                    "Prime exeptionnelle (10%) (DZD)", "Frais de remboursement (Véhicule) (DZD)", 
                    "Indemnité de panier (DZD)", "Indemnité de transport (DZD)", "Nouveau Salaire de base (DZD)",
                    "Prime vestimentaire (DZD)", "Nouvelle Indemnité de panier (DZD)",  "Nouvelle Indemnité de transport (DZD)",
                    "Nouvelle Prime mensuelle (DZD)", "Nouveaux Frais de remboursement (Véhicule) (DZD)","Prime vestimentaire (DZD)", " Indémnité Véhicule (DZD)",
                    "Absence (Jour)","Absence Maladie (Jour)","Absence Maternité (Jour)", "Absence Mise à pied (Jour)", "Jours de congé (Jour)",
                    "Heures supp 100% (H)", "Heures supp 75% (H)", "Heures supp 50% (H)", "Jours supp (Jour)","Taux complément santé (DZD)","Frais téléphone",
                    "Frais de transport (Yassir)","Frais divers","Avance NET (DZD)","Augmentation", "Régul", "Coût congé payé", "Nbr jours STC (jours)",
                    "Jours de congé (22 jours)","Indemnité non cotisable et imposable 10% (DZD)","Indemnité zone", "Total absence (sur 22 jours)",
                    "Nouvelle indémnité Véhicule (DZD)","Nouveau IFSP (20% du salaire de base)","Nbr jours augmentation"
                ]
                for col in cols_to_float:
                    if col in df_client.columns:
                        df_client[col] = nettoyer_colonne(df_client, col)
                    else:
                        df_client[col] = 0.0
                col_pourcentage = ["Fees etalent", "TVA"]

                for col in col_pourcentage:
                    df_client[col] = (
                        df_client[col]
                        .fillna("0")                          
                        .astype(str)
                        .str.replace("%", "", regex=False)
                        .str.replace(",", ".", regex=False)
                        .str.strip()
                        .replace("", "0")
                        .astype(float)
                    )

                df_client["NDF"]= df_client["NDF"].fillna(0)                                 
                # df_client["jours conge ouvres"] = df_client.apply(calcul_jours_ouvres, axis=1)
                
                df_client["jours stc ouvres"] = df_client.apply(calcul_joursstc_ouvres, axis=1)
                absences_total = (
                df_client["Absence (Jour)"]
                + df_client["Absence Maladie (Jour)"]
                + df_client["Absence Maternité (Jour)"]
                + df_client["Absence Mise à pied (Jour)"]
                + df_client["Jours de congé (Jour)"]  # version brute
            )
                print(df_client["Jours de congé (22 jours)"])
                absences_total22 = (
                    df_client["Total absence (sur 22 jours)"]
                    + df_client["Jours de congé (22 jours)"]  # version corrigée week-end
                )
                absences_totallg = (
                df_client["Absence (Jour)"]
                + df_client["Absence Maladie (Jour)"]
                + df_client["Absence Maternité (Jour)"]
                + df_client["Absence Mise à pied (Jour)"]
                
            ) 
                
                HEURES_MOIS = 173.33
                
                # 3. Calculs (une seule fois)
                df_client["Salaire de base calcule"] = (get_valeur("Salaire de base (DZD)", "Nouveau Salaire de base (DZD)"))
                df_client["Indemnité de panier calcule"] = get_valeur("Indemnité de panier (DZD)", "Nouvelle Indemnité de panier (DZD)")
                df_client["indémnité Véhicule calcule"] = get_valeur(" Indémnité Véhicule (DZD)", "Nouvelle indémnité Véhicule (DZD)")
                df_client["Indemnité de transport calcule"] = get_valeur("Indemnité de transport (DZD)", "Nouvelle Indemnité de transport (DZD)")
                df_client["Prime mensuelle calcule"] = get_valeur("Prime mensuelle (DZD)", "Nouvelle Prime mensuelle (DZD)")
                df_client["IFSP (20% du salaire de base) calcule"] = get_valeur("IFSP (20% du salaire de base)", "Nouveau IFSP (20% du salaire de base)")
                df_client["Frais remboursement calcule"] = get_valeur("Frais de remboursement (Véhicule) (DZD)", "Nouveaux Frais de remboursement (Véhicule) (DZD)")
                # print(df_client["Salaire de base calcule"])
                df_client["Salaire de base calcule"] = ((df_client["Salaire de base calcule"]/30)*(30-df_client["Nbr jours augmentation"]))+(((df_client["Salaire de base calcule"] * (1 + (df_client["Augmentation"] / 100)))/30 )*df_client["Nbr jours augmentation"])
                # df_client["Salaire de base calcule"] = (df_client["Salaire de base calcule"] * (1 + (df_client["Augmentation"] / 100)))
                salaire_journalier = df_client["Salaire de base calcule"] / jours_mois
                df_client["Salaire de base calcule"] = (
                    (df_client["Salaire de base calcule"]
                    - df_client["Salaire de base calcule"] / 30 * absences_total
                    + df_client["Salaire de base calcule"] / HEURES_MOIS * (
                        df_client["Heures supp 100% (H)"] * 2
                        + df_client["Heures supp 75% (H)"] * 1.75
                        + df_client["Heures supp 50% (H)"] * 1.5
                    )
                    + (df_client["Jours supp (Jour)"] * salaire_journalier)) +df_client["IFSP (20% du salaire de base) calcule"]
                )
                df_client["IFSP (20% du salaire de base)"] = df_client["IFSP (20% du salaire de base) calcule"]
                df_client["Salaire de base calcule"] = df_client["Salaire de base calcule"] - (df_client["Salaire de base calcule"]/jours_mois) * df_client["Nbr jours STC (jours)"]
                # Ajout régul seulement si Base de régul == "Salaire de base"
                df_client["Salaire de base calcule"] += np.where(
                    df_client["Base de régul"] == "Salaire de base", df_client["Régul"], 0)

                print(absences_total22)
                if df_client["Etablissement"].iloc[0] == "Coca cola": 
                    df_client["Indemnité de panier calcule"] = (
                    df_client["Indemnité de panier calcule"]
                    - (df_client["Indemnité de panier calcule"] / 26 * absences_total22)
                    + (df_client["Indemnité de panier calcule"] / 26 * (
                        (df_client["Heures supp 100% (H)"] ) / 8
                        + (df_client["Heures supp 75% (H)"] ) / 8
                        + (df_client["Heures supp 50% (H)"] ) / 8
                    ))
                )
                    df_client["Indemnité de transport calcule"] = (
                    df_client["Indemnité de transport calcule"]
                    - (df_client["Indemnité de transport calcule"] / 26 * absences_total22)
                    + (df_client["Indemnité de transport calcule"] / 26 * (
                        (df_client["Heures supp 100% (H)"] ) / 8
                        + (df_client["Heures supp 75% (H)"] ) / 8
                        + (df_client["Heures supp 50% (H)"] ) / 8
                    ))
                )
                    df_client["indémnité Véhicule calcule"] = (
                    df_client["indémnité Véhicule calcule"]
                    - (df_client["indémnité Véhicule calcule"] / 26 * absences_total22)
                    + (df_client["indémnité Véhicule calcule"] / 26 * (
                        (df_client["Heures supp 100% (H)"] ) / 8
                        + (df_client["Heures supp 75% (H)"] ) / 8
                        + (df_client["Heures supp 50% (H)"] ) / 8
                    ))
                )
                    df_client["Indemnitésomme"]= df_client["Indemnité de panier calcule"] + df_client["Indemnité de transport calcule"] + df_client["Prime vestimentaire (DZD)"] + df_client["indémnité Véhicule calcule"]+df_client["Avance NET (DZD)"] +df_client["Frais remboursement calcule"]
                    df_client["Indemnité 22jours"] = df_client["Indemnitésomme"]
                    print(df_client["Indemnité 22jours"])
                else:
                    df_client["Indemnitésomme"]= df_client["Indemnité de panier calcule"] + df_client["Indemnité de transport calcule"] + df_client["Prime vestimentaire (DZD)"] + df_client["indémnité Véhicule calcule"]+df_client["Avance NET (DZD)"] +df_client["Frais remboursement calcule"]
                    
                    df_client["Indemnité 22jours"] = (
                        df_client["Indemnitésomme"]
                        - (df_client["Indemnitésomme"] / 22 * absences_total22)
                        + (df_client["Indemnitésomme"] / 22 * (
                            (df_client["Heures supp 100% (H)"] * 2) / 8
                            + (df_client["Heures supp 75% (H)"] * 1.75) / 8
                            + (df_client["Heures supp 50% (H)"] * 1.5) / 8
                        ))
                    )
                df_client["Indemnité 22jours"]= df_client["Indemnité 22jours"] - ((df_client["Indemnité 22jours"]/22) * df_client["jours stc ouvres"])
                if df_client["Etablissement"].iloc[0] == "LG":
                    df_client["Indemnitésomme"]= df_client["Indemnité de panier calcule"] + df_client["Indemnité de transport calcule"] + df_client["Prime vestimentaire (DZD)"] + df_client["indémnité Véhicule calcule"]+df_client["Avance NET (DZD)"] +df_client["Frais remboursement calcule"]
                    
                    df_client["Indemnité 22jours"] = (
                        df_client["Indemnitésomme"]
                        - (df_client["Indemnitésomme"] / 30 * df_client["Total absence (sur 22 jours)"])
                        
                    )
                    df_client["Salaire de base calcule"] = (
                        df_client["Salaire de base (DZD)"]
                        + (df_client["Salaire de base (DZD)"] / HEURES_MOIS * (
                        + df_client["Heures supp 100% (H)"] * 2
                        + df_client["Heures supp 75% (H)"] * 1.75
                        + df_client["Heures supp 50% (H)"] * 1.5))
                        - df_client["Salaire de base (DZD)"] / 30 * absences_totallg 
                    )

                    df_client["Base cotisable"] = df_client["Salaire de base calcule"] + df_client["Prime exeptionnelle (10%) (DZD)"]  + df_client["Prime mensuelle (Barème) (DZD)"]  + df_client["Indemnité non cotisable et imposable 10% (DZD)"]
                else:
                    df_client["Base cotisable"] = (
                            df_client["Prime exeptionnelle (10%) (DZD)"]  + df_client["Indemnité non cotisable et imposable 10% (DZD)"] +
                            df_client["Prime mensuelle calcule"] + df_client["Salaire de base calcule"] + df_client["Prime mensuelle (Barème) (DZD)"] 
                        )
                df_client["indémnité Véhicule"] = df_client["indémnité Véhicule calcule"]
                if df_client["Etablissement"].iloc[0] == "LG" :
                    df_client["Base imposable 10%"] = df_client["Indemnité non cotisable et imposable 10% (DZD)"] * 0.91
                else:
                    df_client["Base imposable 10%"] = df_client["Indemnité non cotisable et imposable 10% (DZD)"] * 0.91

                df_client["Retenue CNAS employé"] = df_client["Base cotisable"] * 0.09
                if df_client["Etablissement"].iloc[0] == "Henkel": 
                    
                    df_client["Base imposable au baréme"]  = ((((df_client["Salaire de base calcule"]+ df_client["Prime mensuelle calcule"])-((df_client["Salaire de base calcule"]+ df_client["Prime mensuelle calcule"])*0.09))+df_client["Indemnité 22jours"])/10)*10
                elif   df_client["Etablissement"].iloc[0] == "LG":
                        df_client["Base imposable au baréme"] = np.floor(((((df_client["Salaire de base calcule"] +df_client["Prime exeptionnelle (10%) (DZD)"] ) * 0.91)+ df_client["Indemnité 22jours"]))/ 10) * 10
                elif df_client["Etablissement"].iloc[0] == "G+D":
                    df_client["Base imposable au baréme"] = np.floor((((df_client["Salaire de base calcule"] + df_client["Indemnité non cotisable et imposable 10% (DZD)"]) -df_client["Indemnité non cotisable et imposable 10% (DZD)"]) * 0.91 + df_client["Indemnité 22jours"])/10)*10
                else:
                    df_client["Base imposable au baréme"] = np.floor((((df_client["Base cotisable"] - df_client["Prime exeptionnelle (10%) (DZD)"]- df_client["Indemnité non cotisable et imposable 10% (DZD)"]- df_client["Indemnité zone"]) * 0.91+ (df_client["Indemnité 22jours"])-df_client["Frais remboursement calcule"]))/ 10) * 10
                    
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
                if df_client["Etablissement"].iloc[0] == "LG":
                    df_client["Salaire brut"] = (
                        (df_client["Salaire de base calcule"] +
                        df_client["Indemnité 22jours"]+
                        df_client["Indemnité non cotisable et imposable 10% (DZD)"]) 
                    )
                    df_client["Salaire brut"] += np.where(
                    df_client["Base de régul"] == "Salaire Brut", df_client["Régul"], 0)
                    df_client["Salaire net"] = ((df_client["Salaire de base calcule"]*0.91)+df_client["Indemnité 22jours"])+df_client["Base imposable 10%"]-df_client["IRG barème"]-df_client["IRG 10%"]
                    df_client["Salaire net"] += np.where(
                    df_client["Base de régul"] == "Salaire Net", df_client["Régul"], 0).round(0)
               
                else : 
                    df_client["Salaire brut"] = (
                        (df_client["Base cotisable"] +
                        (df_client["Indemnité 22jours"]- df_client["Frais remboursement calcule"])+
                        df_client["Frais remboursement calcule"]) 
                    )
                    df_client["Salaire brut"] += np.where(
                    df_client["Base de régul"] == "Salaire Brut", df_client["Régul"], 0)
                    df_client["Salaire net"] = (
                        (df_client["Salaire brut"] -
                        df_client["Retenue CNAS employé"] -
                        df_client["IRG barème"] -
                        df_client["IRG 10%"]) 
                    )
                    df_client["Salaire net"] += np.where(
                    df_client["Base de régul"] == "Salaire Net", df_client["Régul"], 0).round(0)
                df_client["CNAS employeur"] = df_client["Base cotisable"] * 0.26
                df_client["Indemnités Non Cotisable - Mensuelle | Panier, Transport"] = df_client["Indemnité 22jours"]
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
                    df_client["Taxe formation et os"] = 0
                    df_client["Masse salariale"] = (
                    df_client["Salaire brut"] +
                    df_client["CNAS employeur"] +
                    df_client["Cotisation œuvre sociale"] +
                    df_client["Taxe formation"]
                )
                df_client["Prime mensuelle"] = df_client["Prime mensuelle calcule"]
                if df_client["Etablissement"].iloc[0] == "Henkel":
                    df_client["Coût salaire"] = (
                        (df_client["Salaire net"]
                        + df_client["Taxe formation et os"]
                        + df_client["CNAS employeur"]
                        + df_client["IRG 10%"]
                        + df_client["IRG barème"]
                        + df_client["Retenue CNAS employé"]
                        + df_client["Frais téléphone"])
                    )
                    df_client["Coût salaire"] += np.where(
                    df_client["Base de régul"] == "Cout salaire", df_client["Régul"], 0)
                    if df_client["Augmentation state"].iloc[0] == "Yes" :

                        df_client["Coût congé payé"] = (df_client["Coût salaire"] / 30 * 2.5) 
                        df_client["Coût congé payé"] += np.where(
                        df_client["Base de régul"] == "Congé payé", df_client["Régul"], 0)
                        fees_multiplicateur = 1 + (df_client["Fees etalent"] / 100)
                    else :
                        df_client["Coût congé payé"] = df_client["Coût congé payé"] 
                        df_client["Coût congé payé"] += np.where(
                        df_client["Base de régul"] == "Congé payé", df_client["Régul"], 0)
                        fees_multiplicateur = 1 + (df_client["Fees etalent"] / 100)
                    
                    
                    if df_client["TAP"].iloc[0] == "Oui" :
                        df_client["TAP (DZD)"] = (df_client["Coût salaire"]+ df_client["Coût congé payé"] + df_client["Taux complément santé (DZD)"])*0.03
                        df_client["Facture HT"] = ((df_client["Coût salaire"] + df_client["Coût congé payé"]+ df_client["TAP (DZD)"] + df_client["Taux complément santé (DZD)"]) * fees_multiplicateur)
                        df_client["Facture HT + NDF"] = df_client["Facture HT"]+df_client["NDF"]
                        df_client["Facture HT + NDF"] = pd.to_numeric(df_client["Facture HT + NDF"], errors="coerce").fillna(0)

                    else : 
                        df_client["TAP (DZD)"] = 0.0
                        df_client["Facture HT"] = ((df_client["Coût salaire"] + df_client["Coût congé payé"]+ df_client["TAP (DZD)"])* fees_multiplicateur)
                        df_client["Facture HT + NDF"] = df_client["Facture HT"]+df_client["NDF"]
                        df_client["Facture HT + NDF"] = pd.to_numeric(df_client["Facture HT + NDF"], errors="coerce").fillna(0)

                elif df_client["Etablissement"].iloc[0] == "LG":
                    df_client["Coût salaire"] = (
                        (df_client["Salaire net"]
                        + df_client["Cotisation œuvre sociale"]
                        + df_client["Taxe formation"]
                        + df_client["CNAS employeur"]
                        + df_client["IRG 10%"]
                        + df_client["IRG barème"]
                        + df_client["Retenue CNAS employé"]
                        + df_client["Frais téléphone"]) 
                    )
                    df_client["Coût salaire"] += np.where(
                    df_client["Base de régul"] == "Cout salaire", df_client["Régul"], 0)
                    if df_client["Augmentation state"].iloc[0] == "Yes" :

                        df_client["Coût congé payé"] = (df_client["Coût salaire"] / 30 * 2.5) 
                        df_client["Coût congé payé"] += np.where(
                        df_client["Base de régul"] == "Congé payé", df_client["Régul"], 0)
                        fees_multiplicateur = 1 + (df_client["Fees etalent"] / 100)
                    else :
                        df_client["Coût congé payé"] = df_client["Coût congé payé"] 
                        df_client["Coût congé payé"] += np.where(
                        df_client["Base de régul"] == "Congé payé", df_client["Régul"], 0)
                        fees_multiplicateur = 1 + (df_client["Fees etalent"] / 100)
                    fees_multiplicateur = 1 + (df_client["Fees etalent"] / 100)
                    if df_client["TAP"].iloc[0] == "Oui" :
                        df_client["TAP (DZD)"] = (df_client["Coût salaire"] + ( df_client["Coût salaire"] * df_client["Fees etalent"])) * 0.02
                        df_client["Facture HT"] = ((df_client["Coût salaire"] * fees_multiplicateur) + df_client["TAP (DZD)"])+ df_client["Taux complément santé (DZD)"] 
                        df_client["Facture HT + NDF"] = df_client["Facture HT"]+df_client["NDF"]
                    else : 
                        df_client["TAP (DZD)"] = 0.0
                        df_client["Facture HT"] = ((df_client["Coût salaire"] * fees_multiplicateur))+ df_client["Taux complément santé (DZD)"] 
                        df_client["Facture HT + NDF"] = df_client["Facture HT"]+df_client["NDF"]
                        df_client["Facture HT + NDF"] = pd.to_numeric(df_client["Facture HT + NDF"], errors="coerce").fillna(0)

                elif df_client["Etablissement"].iloc[0] == "Maersk":
                    df_client["Coût salaire"] = (
                        (df_client["Salaire net"]
                        + df_client["Taxe formation et os"]
                        + df_client["CNAS employeur"]
                        + df_client["IRG 10%"]
                        + df_client["IRG barème"]
                        + df_client["Retenue CNAS employé"])
                    )
                    df_client["Coût salaire"] += np.where(
                    df_client["Base de régul"] == "Cout salaire", df_client["Régul"], 0)
                    if df_client["Augmentation state"].iloc[0] == "Yes" :

                        df_client["Coût congé payé"] = (df_client["Coût salaire"] / 30 * 2.5) 
                        df_client["Coût congé payé"] += np.where(
                        df_client["Base de régul"] == "Congé payé", df_client["Régul"], 0)
                        fees_multiplicateur = 1 + (df_client["Fees etalent"] / 100)
                    else :
                        df_client["Coût congé payé"] = df_client["Coût congé payé"] 
                        df_client["Coût congé payé"] += np.where(
                        df_client["Base de régul"] == "Congé payé", df_client["Régul"], 0)
                        fees_multiplicateur = 1 + (df_client["Fees etalent"] / 100)
                    fees_multiplicateur = 1 + (df_client["Fees etalent"] / 100)
                    if df_client["TAP"].iloc[0] == "Oui" :
                        df_client["TAP (DZD)"] = (df_client["Coût salaire"] + ( df_client["Coût salaire"] * df_client["Fees etalent"])) * 0.02
                        df_client["Facture HT"] = ((df_client["Coût salaire"] + df_client["Coût congé payé"]+ df_client["TAP (DZD)"])* fees_multiplicateur)
                        df_client["Facture HT + NDF"] = df_client["Facture HT"]+df_client["NDF"]
                        df_client["Facture HT + NDF"] = pd.to_numeric(df_client["Facture HT + NDF"], errors="coerce").fillna(0)

                    else : 
                        df_client["TAP (DZD)"] = 0.0
                        df_client["Facture HT"] = ((df_client["Coût salaire"] + df_client["Coût congé payé"]+ df_client["TAP (DZD)"])* fees_multiplicateur)
                        df_client["Facture HT + NDF"] = df_client["Facture HT"]+df_client["NDF"]
                elif df_client["Etablissement"].iloc[0] == "G+D":
                    df_client["Coût salaire"] = (df_client["Salaire de base calcule"] + df_client["Indemnité de panier calcule"] + df_client["Indemnité de transport calcule"] +df_client["Prime vestimentaire (DZD)"]+df_client["Frais remboursement calcule"]+df_client["Prime exeptionnelle (10%) (DZD)"]) 
                    df_client["Coût salaire"] += np.where(
                    df_client["Base de régul"] == "Cout salaire", df_client["Régul"], 0)
                    fees_multiplicateur = 1 + (df_client["Fees etalent"] / 100)
                    if df_client["TAP"].iloc[0] == "Oui" :
                        df_client["TAP (DZD)"] = (df_client["Coût salaire"] + ( df_client["Coût salaire"] * df_client["Fees etalent"])) * 0.02
                        df_client["Facture HT"] = ((df_client["Coût salaire"] * fees_multiplicateur) + df_client["TAP (DZD)"])
                        df_client["Facture HT + NDF"] = df_client["Facture HT"]+df_client["NDF"]
                        df_client["Facture HT + NDF"] = pd.to_numeric(df_client["Facture HT + NDF"], errors="coerce").fillna(0)

                    else : 
                        df_client["TAP (DZD)"] = 0.0
                        df_client["Facture HT"] = ((df_client["Coût salaire"] * fees_multiplicateur))
                        df_client["Facture HT + NDF"] = df_client["Facture HT"]+df_client["NDF"]
                        df_client["Facture HT + NDF"] = pd.to_numeric(df_client["Facture HT + NDF"], errors="coerce").fillna(0)

                else:
                    if df_client["Augmentation state"].iloc[0] == "Yes" :

                        df_client["Coût congé payé"] = (df_client["Coût salaire"] / 30 * 2.5) 
                        df_client["Coût congé payé"] += np.where(
                        df_client["Base de régul"] == "Congé payé", df_client["Régul"], 0)
                        fees_multiplicateur = 1 + (df_client["Fees etalent"] / 100)
                    else :
                        df_client["Coût congé payé"] = df_client["Coût congé payé"] 
                        df_client["Coût congé payé"] += np.where(
                        df_client["Base de régul"] == "Congé payé", df_client["Régul"], 0)
                        fees_multiplicateur = 1 + (df_client["Fees etalent"] / 100)
                    df_client["Coût salaire"] = (
                        (df_client["Masse salariale"]
                        + df_client["Coût congé payé"]
                        + df_client["Taux complément santé (DZD)"]
                        + df_client["Frais divers"]
                        + df_client["Frais de transport (Yassir)"]
                        + df_client["Frais téléphone"]) 
                    )
                    df_client["Coût salaire"] += np.where(
                    df_client["Base de régul"] == "Cout salaire", df_client["Régul"], 0)
                    fees_multiplicateur = 1 + (df_client["Fees etalent"] / 100)
                    if df_client["TAP"].iloc[0] == "Oui" :
                        df_client["TAP (DZD)"] = (df_client["Coût salaire"] + ( df_client["Coût salaire"] * (df_client["Fees etalent"]/100))) * 0.02
                        df_client["Facture HT"] = ((df_client["Coût salaire"] * fees_multiplicateur) + df_client["TAP (DZD)"])
                        df_client["Facture HT + NDF"] = df_client["Facture HT"] + df_client["NDF"]
                        df_client["Facture HT + NDF"] = pd.to_numeric(df_client["Facture HT + NDF"], errors="coerce").fillna(0)

                    else : 
                        df_client["TAP (DZD)"] = 0.0
                        df_client["Facture HT"] = ((df_client["Coût salaire"] * fees_multiplicateur))
                        df_client["Facture HT + NDF"] = df_client["Facture HT"] + df_client["NDF"]
                        df_client["Facture HT + NDF"] = pd.to_numeric(df_client["Facture HT + NDF"], errors="coerce").fillna(0)
                df_client["Frais remboursement"] = df_client["Frais remboursement calcule"]
                df_client["Salaire de base"] = df_client["Salaire de base calcule"]
                df_client["Indemnité de panier"] = df_client["Indemnité de panier calcule"]
                df_client["Indemnité de transport"] = df_client["Indemnité de transport calcule"]
                tva_multiplicateur = 1+ (df_client["TVA"]/100)
                # Calcul TVA et TTC
                df_client["Facture TVA"] = df_client["Facture HT + NDF"] * (df_client["TVA"] / 100)
                df_client["Facture TTC"] = df_client["Facture HT + NDF"] + df_client["Facture TVA"]
                df_client["Observation"] = Observation
                st.write(df_client.head(50)) # On peut encapsuler ton code de calculs dans une fonction
                
                # 1. On définit les colonnes fixes (identité employé)
                # Colonnes fixes (identité employé)
                mois_ordre = ["Janvier": "-janv.-", "Février":"-févr.-", "Mars":"-mars-", "Avril":"-avr.-", "Mai":"-mai-", "Juin":"-juin-",
              "Juillet": "-juil.-", "Août":"-août-", "Septembre":"-sept.-", "Octobre":"-oct.-", "Novembre":"-nov.-", "Décembre":"-déc.-"]
                id_cols = ["Nom", "Prénom", "N°", "Titre du poste", "Durée du CDD (Mois)", "Etablissement", "Année"]

                # Colonnes variables (salaire, primes…)
                val_cols = [c for c in df_client.columns if c not in id_cols + ["Mois"]]

                # Pivot : Mois devient colonnes
                df_pivot = df_client.pivot_table(
                    index=id_cols,
                    columns="Mois",
                    values=val_cols,
                    aggfunc="first"
                )

                # Aplatir les colonnes multi-index
                df_pivot.columns = [f"{val}_{mois}" for val, mois in df_pivot.columns]
                df_pivot = df_pivot.reset_index()

                # Réordonner les colonnes mois
                colonnes_identite = id_cols
                colonnes_mois = []

                for mois in mois_ordre:
                    colonnes_mois.extend([c for c in df_pivot.columns if c.endswith(f"_{mois}")])

                # Appliquer le nouvel ordre
                df_pivot = df_pivot[colonnes_identite + colonnes_mois]

               
                # 📥 Génération et téléchargement Excel
                # --------------------------------------
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_client.to_excel(writer, index=False, sheet_name='Calculs')
                    workbook = writer.book
                    worksheet = writer.sheets['Calculs']

                    # Style entête
                    header_format = workbook.add_format({
                        'bold': True, 'text_wrap': True, 'valign': 'middle',
                        'align': 'center', 'fg_color': '#0a5275',
                        'font_color': 'white', 'border': 1
                    })
                    for col_num, value in enumerate(df_client.columns.values):
                        worksheet.write(0, col_num, value, header_format)

                    # Ajuster largeur colonnes
                    for i, col in enumerate(df_client.columns):
                        col_width = max(df_client[col].astype(str).map(len).max(), len(col)) + 2
                        worksheet.set_column(i, i, col_width)

                st.download_button(
                    label="📊 Télécharger les résultats en Excel",
                    data=output.getvalue(),
                    file_name=f"{st.session_state.selected_client}_calculs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

               
                # 📥 Génération et téléchargement PDF par employé
                # ------------------------------------------------
                st.markdown("### 📥 Télécharger la facture PDF par employé")
                for idx, row in df_pivot.iterrows():
                    nom = str(row.get("Nom", f"employe_{idx}")).strip().replace(" ", "_")
                    matricule = str(row.get("Matricule", f"id_{idx}")).strip()

                    employe_data = row.to_dict()

                    # Générer UN SEUL fichier consolidé avec tous les mois
                    fichier_excel = generer_facture_excel(employe_data, f"{matricule}_{nom}_facture.xlsx")

                    with open(fichier_excel, "rb") as f:
                        excel_data = f.read()

                    st.download_button(
                        label=f"📊 {nom}",
                        data=excel_data,
                        file_name=f"{nom}_facture.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_{matricule}_{idx}"
                    )

                    import os
                    if os.path.exists(fichier_excel):
                        os.remove(fichier_excel)


            else:
                st.warning("⚠️ Aucun employé trouvé pour ce client ")
        else:
            st.info("Veuillez d'abord téléverser le fichier récapitulatif global dans la barre latérale.")














