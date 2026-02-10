"""
ChemicalClassifier - классификатор соединений для ChronobioticsDB.
Классифицирует соединения по структурным классам, фармакологической активности и биологическим эффектам.
Интегрирован с моделями Django из main/models.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
import logging
from collections import Counter, defaultdict

# Химические библиотеки
try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem,
        Descriptors,
        Lipinski,
        Crippen,
        Fragments,
        rdMolDescriptors,
    )
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    from rdkit.Chem import rdFingerprintGenerator

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit не установлен. Некоторые функции будут недоступны.")

# ML библиотеки
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.multilabel import MultiOutputClassifier
    import xgboost as xgb

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Scikit-learn не установлен. ML классификация будет недоступна.")

from django.conf import settings
from django.db import transaction, connection
from django.core.exceptions import ObjectDoesNotExist
from django.apps import apps

# Импорт моделей из main/models.py
try:
    from main.models import (
        Chronobiotic,
        Bioclass,
        Effect,
        Mechanism,
        Targets,
        Articles,
        Synonyms,
    )

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logging.warning("Модели Django не найдены. Работа с БД будет недоступна.")

logger = logging.getLogger(__name__)


class ChemicalClassifier:
    """
    Классификатор химических соединений для ChronobioticsDB.

    Обеспечивает классификацию по:
    1. Структурным классам (терпены, стероиды, алкалоиды и др.)
    2. Фармакологической активности (через механизмы и эффекты)
    3. Биологическим эффектам
    """

    # Структурные классы из описания ChronobioticsDB
    STRUCTURAL_CLASSES = [
        "Терпены",
        "Стероиды",
        "Алкалоиды",
        "Сульфаниламиды",
        "Сульфонаты",
        "Ароматические соединения (фенолы и флавоноиды)",
        "Фторорганические",
        "Хлорорганические соединения",
        "Четвертичные аммониевые соединения",
        "Амиды",
        "Пептиды",
        "Сложные эфиры",
        "Липиды",
        "Гетероциклы (пиридин и тиофен)",
        "Нитросоединения",
        "Фосфаты",
        "Металлические соли органических веществ",
    ]

    # Группы фармакологической активности на основе механизмов
    PHARMACOLOGICAL_ACTIVITY_GROUPS = {
        "Нейромодуляторы": [
            "мелатонин",
            "серотонин",
            "дофамин",
            "ГАМК",
            "глутамат",
            "нейротрансмиттер",
        ],
        "Гормональные модуляторы": ["кортизол", "гормон", "стероид", "эндокринный"],
        "Антиоксиданты": ["антиоксидант", "окислительный стресс", "свободные радикалы"],
        "Противовоспалительные": ["воспаление", "цитокин", "простагландин"],
        "Циркадные модуляторы": ["циркадный", "ритм", "часы", "суточный"],
        "Метаболические модуляторы": ["метаболизм", "фермент", "метаболит"],
        "Иммуномодуляторы": ["иммунный", "иммунитет", "лимфоцит"],
        "Антимикробные": ["антибактериальный", "антивирусный", "противогрибковый"],
    }

    def __init__(self, model_dir: str = None):
        """
        Инициализация классификатора.

        Args:
            model_dir: Директория для сохранения/загрузки моделей
        """
        self.model_dir = model_dir or os.path.join(
            settings.BASE_DIR, "models", "chemical_classifier"
        )
        os.makedirs(self.model_dir, exist_ok=True)

        # Модели классификации
        self.structural_model = None
        self.activity_model = None
        self.effect_model = None

        # Кодировщики меток
        self.label_encoders = {}

        # Мультилейбл кодировщики
        self.multilabel_binarizers = {}

        # Скейлеры
        self.scalers = {}

        # Фичи для классификации
        self.feature_names = []

        # Инициализация путей к моделям
        self.model_paths = {
            "structural": os.path.join(self.model_dir, "structural_classifier.pkl"),
            "activity": os.path.join(self.model_dir, "activity_classifier.pkl"),
            "effect": os.path.join(self.model_dir, "effect_classifier.pkl"),
            "encoders": os.path.join(self.model_dir, "label_encoders.pkl"),
            "binarizers": os.path.join(self.model_dir, "multilabel_binarizers.pkl"),
            "scalers": os.path.join(self.model_dir, "scalers.pkl"),
            "features": os.path.join(self.model_dir, "feature_names.pkl"),
            "statistics": os.path.join(self.model_dir, "classifier_statistics.json"),
        }

        # Попытка загрузить существующие модели
        self.load_models()

        # Статистика классификации
        self.stats = {
            "total_classified": 0,
            "last_training": None,
            "last_database_update": None,
            "accuracy": {},
            "coverage": {},
        }

        logger.info(
            f"ChemicalClassifier инициализирован. Директория моделей: {self.model_dir}"
        )

    def load_models(self) -> bool:
        """
        Загрузка предварительно обученных моделей.

        Returns:
            bool: Успешна ли загрузка
        """
        try:
            if os.path.exists(self.model_paths["structural"]):
                self.structural_model = joblib.load(self.model_paths["structural"])
                logger.info("Загружена модель структурной классификации")

            if os.path.exists(self.model_paths["activity"]):
                self.activity_model = joblib.load(self.model_paths["activity"])
                logger.info(
                    "Загружена модель классификации фармакологической активности"
                )

            if os.path.exists(self.model_paths["effect"]):
                self.effect_model = joblib.load(self.model_paths["effect"])
                logger.info("Загружена модель классификации биологических эффектов")

            if os.path.exists(self.model_paths["encoders"]):
                self.label_encoders = joblib.load(self.model_paths["encoders"])
                logger.info("Загружены кодировщики меток")

            if os.path.exists(self.model_paths["binarizers"]):
                self.multilabel_binarizers = joblib.load(self.model_paths["binarizers"])
                logger.info("Загружены мультилейбл кодировщики")

            if os.path.exists(self.model_paths["scalers"]):
                self.scalers = joblib.load(self.model_paths["scalers"])
                logger.info("Загружены скейлеры")

            if os.path.exists(self.model_paths["features"]):
                self.feature_names = joblib.load(self.model_paths["features"])
                logger.info("Загружены имена фич")

            if os.path.exists(self.model_paths["statistics"]):
                with open(self.model_paths["statistics"], "r") as f:
                    self.stats.update(json.load(f))
                logger.info("Загружена статистика классификатора")

            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            return False

    def save_models(self) -> bool:
        """
        Сохранение обученных моделей.

        Returns:
            bool: Успешно ли сохранение
        """
        try:
            if self.structural_model:
                joblib.dump(self.structural_model, self.model_paths["structural"])

            if self.activity_model:
                joblib.dump(self.activity_model, self.model_paths["activity"])

            if self.effect_model:
                joblib.dump(self.effect_model, self.model_paths["effect"])

            if self.label_encoders:
                joblib.dump(self.label_encoders, self.model_paths["encoders"])

            if self.multilabel_binarizers:
                joblib.dump(self.multilabel_binarizers, self.model_paths["binarizers"])

            if self.scalers:
                joblib.dump(self.scalers, self.model_paths["scalers"])

            if self.feature_names:
                joblib.dump(self.feature_names, self.model_paths["features"])

            # Сохранение статистики
            with open(self.model_paths["statistics"], "w") as f:
                json.dump(self.stats, f, indent=2)

            logger.info("Модели успешно сохранены")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения моделей: {e}")
            return False

    def extract_features_from_smiles(self, smiles: str) -> Optional[Dict[str, float]]:
        """
        Извлечение химических дескрипторов из SMILES.

        Args:
            smiles: SMILES строка

        Returns:
            Dict: Словарь дескрипторов или None при ошибке
        """
        if not RDKIT_AVAILABLE:
            logger.error("RDKit не доступен для извлечения фич")
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Не удалось разобрать SMILES: {smiles}")
                return None

            features = {}

            # Базовые физико-химические свойства
            features["mol_weight"] = Descriptors.MolWt(mol)
            features["logp"] = Crippen.MolLogP(mol)
            features["tpsa"] = Descriptors.TPSA(mol)
            features["num_h_donors"] = Lipinski.NumHDonors(mol)
            features["num_h_acceptors"] = Lipinski.NumHAcceptors(mol)
            features["num_rotatable_bonds"] = Lipinski.NumRotatableBonds(mol)
            features["num_aromatic_rings"] = Lipinski.NumAromaticRings(mol)
            features["num_aliphatic_rings"] = Lipinski.NumAliphaticRings(mol)
            features["num_saturated_rings"] = Lipinski.NumSaturatedRings(mol)
            features["num_heteroatoms"] = Lipinski.NumHeteroatoms(mol)
            features["num_heavy_atoms"] = Lipinski.HeavyAtomCount(mol)
            features["num_valence_electrons"] = Descriptors.NumValenceElectrons(mol)

            # Количества атомов
            atom_counts = defaultdict(int)
            for atom in mol.GetAtoms():
                atom_counts[atom.GetSymbol()] += 1

            features["num_atoms"] = mol.GetNumAtoms()
            features["num_bonds"] = mol.GetNumBonds()
            features["num_c"] = atom_counts.get("C", 0)
            features["num_h"] = atom_counts.get("H", 0)
            features["num_o"] = atom_counts.get("O", 0)
            features["num_n"] = atom_counts.get("N", 0)
            features["num_s"] = atom_counts.get("S", 0)
            features["num_p"] = atom_counts.get("P", 0)
            features["num_f"] = atom_counts.get("F", 0)
            features["num_cl"] = atom_counts.get("Cl", 0)
            features["num_br"] = atom_counts.get("Br", 0)
            features["num_i"] = atom_counts.get("I", 0)

            # Процентный состав
            if features["num_atoms"] > 0:
                for elem in ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"]:
                    features[f"percent_{elem.lower()}"] = (
                        atom_counts.get(elem, 0) / features["num_atoms"]
                    ) * 100

            # Функциональные группы
            features["num_oh"] = Fragments.fr_Al_OH(mol) + Fragments.fr_Ar_OH(mol)
            features["num_cooh"] = Fragments.fr_COO(mol)
            features["num_nh2"] = Fragments.fr_NH2(mol)
            features["num_nh"] = Fragments.fr_NH(mol)
            features["num_no2"] = Fragments.fr_NO2(mol)
            features["num_po4"] = Fragments.fr_phos_acid(mol) + Fragments.fr_phos_ester(
                mol
            )
            features["num_so2"] = Fragments.fr_sulfonamd(mol) + Fragments.fr_sulfone(
                mol
            )
            features["num_cn"] = Fragments.fr_C_N(mol)
            features["num_nc"] = Fragments.fr_N_O(mol)
            features["num_co"] = Fragments.fr_C_O(mol) + Fragments.fr_C_O_noCOO(mol)
            features["num_cs"] = Fragments.fr_C_S(mol)
            features["num_halogens"] = (
                features["num_f"]
                + features["num_cl"]
                + features["num_br"]
                + features["num_i"]
            )

            # Индексы и дескрипторы
            features["balaban_j"] = Descriptors.BalabanJ(mol)
            features["bertz_ct"] = Descriptors.BertzCT(mol)
            features["hall_kier_alpha"] = Descriptors.HallKierAlpha(mol)
            features["kappa1"] = Descriptors.Kappa1(mol)
            features["kappa2"] = Descriptors.Kappa2(mol)
            features["kappa3"] = Descriptors.Kappa3(mol)

            # Фингерпринты (первые 100 битов)
            fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=100)
            for i in range(100):
                features[f"fp_{i}"] = fp[i]

            # Ринг-фингерпринты
            ring_fp = rdMolDescriptors.GetHashedRingFingerprint(mol)
            for i in range(min(20, len(ring_fp))):
                features[f"ring_fp_{i}"] = ring_fp[i]

            return features

        except Exception as e:
            logger.error(f"Ошибка извлечения фич из SMILES {smiles}: {e}")
            return None

    def get_existing_classes_from_db(self) -> Dict[str, List[str]]:
        """
        Получение существующих классов, механизмов и эффектов из БД.

        Returns:
            Dict: Словарь с существующими категориями
        """
        if not MODELS_AVAILABLE:
            return {"structural_classes": [], "mechanisms": [], "effects": []}

        try:
            # Получение структурных классов
            structural_classes = list(
                Bioclass.objects.values_list("nameclass", flat=True)
            )

            # Получение механизмов
            mechanisms = list(Mechanism.objects.values_list("mechanismname", flat=True))

            # Получение эффектов
            effects = list(Effect.objects.values_list("Effectname", flat=True))

            return {
                "structural_classes": structural_classes,
                "mechanisms": mechanisms,
                "effects": effects,
            }
        except Exception as e:
            logger.error(f"Ошибка получения классов из БД: {e}")
            return {"structural_classes": [], "mechanisms": [], "effects": []}

    def prepare_training_data_from_db(self) -> Tuple:
        """
        Подготовка данных для обучения из базы данных.

        Returns:
            Tuple: (X, y_structural, y_activity, y_effect)
        """
        if not MODELS_AVAILABLE:
            logger.error("Модели Django не доступны")
            return np.array([]), [], [], []

        try:
            # Получение всех соединений с их классами, механизмами и эффектами
            compounds = Chronobiotic.objects.prefetch_related(
                "classf", "mechanisms", "effect"
            ).all()

            features = []
            structural_labels = []
            mechanism_labels = []
            effect_labels = []

            for compound in compounds:
                smiles = compound.smiles
                if not smiles:
                    continue

                # Извлечение фич
                feats = self.extract_features_from_smiles(smiles)
                if not feats:
                    continue

                features.append(list(feats.values()))

                # Структурные классы (множественный выбор)
                structural_classes = list(
                    compound.classf.values_list("nameclass", flat=True)
                )
                structural_labels.append(
                    structural_classes if structural_classes else ["Неизвестно"]
                )

                # Механизмы (для фармакологической активности)
                mechanisms = list(
                    compound.mechanisms.values_list("mechanismname", flat=True)
                )
                mechanism_labels.append(mechanisms if mechanisms else ["Неизвестно"])

                # Эффекты
                effects = list(compound.effect.values_list("Effectname", flat=True))
                effect_labels.append(effects if effects else ["Неизвестно"])

            if not features:
                logger.warning("Нет данных для обучения")
                return np.array([]), [], [], []

            X = np.array(features)

            # Сохранение имен фич
            if feats:
                self.feature_names = list(feats.keys())

            return X, structural_labels, mechanism_labels, effect_labels

        except Exception as e:
            logger.error(f"Ошибка подготовки данных из БД: {e}")
            return np.array([]), [], [], []

    def train_models_from_database(self, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Обучение моделей на данных из базы данных.

        Args:
            test_size: Доля тестовых данных

        Returns:
            Dict: Метрики и информация об обучении
        """
        if not ML_AVAILABLE:
            logger.error("ML библиотеки не доступны для обучения")
            return {}

        try:
            # Подготовка данных
            X, y_struct, y_mech, y_effect = self.prepare_training_data_from_db()

            if len(X) == 0:
                logger.warning("Нет данных для обучения в базе данных")
                return {}

            logger.info(f"Подготовлено {len(X)} образцов для обучения")

            # Мультилейбл кодирование
            self.multilabel_binarizers["structural"] = MultiLabelBinarizer()
            self.multilabel_binarizers["mechanism"] = MultiLabelBinarizer()
            self.multilabel_binarizers["effect"] = MultiLabelBinarizer()

            y_struct_binary = self.multilabel_binarizers["structural"].fit_transform(
                y_struct
            )
            y_mech_binary = self.multilabel_binarizers["mechanism"].fit_transform(
                y_mech
            )
            y_effect_binary = self.multilabel_binarizers["effect"].fit_transform(
                y_effect
            )

            # Разделение на train/test
            X_train, X_test, y_struct_train, y_struct_test = train_test_split(
                X, y_struct_binary, test_size=test_size, random_state=42
            )

            _, _, y_mech_train, y_mech_test = train_test_split(
                X, y_mech_binary, test_size=test_size, random_state=42
            )

            _, _, y_effect_train, y_effect_test = train_test_split(
                X, y_effect_binary, test_size=test_size, random_state=42
            )

            # Масштабирование признаков
            self.scalers["structural"] = StandardScaler()
            self.scalers["mechanism"] = StandardScaler()
            self.scalers["effect"] = StandardScaler()

            X_train_struct = self.scalers["structural"].fit_transform(X_train)
            X_train_mech = self.scalers["mechanism"].fit_transform(X_train)
            X_train_effect = self.scalers["effect"].fit_transform(X_train)

            # Обучение моделей

            # 1. Модель для структурных классов (мультилейбл)
            logger.info("Обучение модели структурной классификации...")
            self.structural_model = MultiOutputClassifier(
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight="balanced",
                )
            )
            self.structural_model.fit(X_train_struct, y_struct_train)

            # 2. Модель для механизмов (фармакологической активности)
            logger.info("Обучение модели классификации механизмов...")
            self.activity_model = MultiOutputClassifier(
                GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
                )
            )
            self.activity_model.fit(X_train_mech, y_mech_train)

            # 3. Модель для эффектов
            logger.info("Обучение модели классификации эффектов...")
            self.effect_model = MultiOutputClassifier(
                xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            )
            self.effect_model.fit(X_train_effect, y_effect_train)

            # Оценка моделей
            metrics = {}

            # Структурная классификация
            X_test_struct = self.scalers["structural"].transform(X_test)
            y_struct_pred = self.structural_model.predict(X_test_struct)
            metrics["structural_f1"] = f1_score(
                y_struct_test, y_struct_pred, average="micro"
            )
            metrics["structural_precision"] = f1_score(
                y_struct_test, y_struct_pred, average="micro"
            )
            metrics["structural_recall"] = f1_score(
                y_struct_test, y_struct_pred, average="micro"
            )

            # Механизмы
            X_test_mech = self.scalers["mechanism"].transform(X_test)
            y_mech_pred = self.activity_model.predict(X_test_mech)
            metrics["mechanism_f1"] = f1_score(
                y_mech_test, y_mech_pred, average="micro"
            )

            # Эффекты
            X_test_effect = self.scalers["effect"].transform(X_test)
            y_effect_pred = self.effect_model.predict(X_test_effect)
            metrics["effect_f1"] = f1_score(
                y_effect_test, y_effect_pred, average="micro"
            )

            # Сохранение моделей
            self.save_models()

            # Обновление статистики
            self.stats["last_training"] = datetime.now().isoformat()
            self.stats["accuracy"] = metrics
            self.stats["training_samples"] = len(X_train)
            self.stats["test_samples"] = len(X_test)
            self.stats["total_features"] = len(self.feature_names)

            # Информация о классах
            db_classes = self.get_existing_classes_from_db()
            self.stats["available_classes"] = {
                "structural": len(db_classes["structural_classes"]),
                "mechanisms": len(db_classes["mechanisms"]),
                "effects": len(db_classes["effects"]),
            }

            logger.info(f"Модели обучены. Метрики: {metrics}")

            return {
                "status": "success",
                "metrics": metrics,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(self.feature_names),
                "available_classes": self.stats["available_classes"],
            }

        except Exception as e:
            logger.error(f"Ошибка обучения моделей: {e}")
            return {"status": "error", "message": str(e)}

    def predict_compound_classes(self, smiles: str) -> Dict[str, Any]:
        """
        Предсказание классов для одного соединения.

        Args:
            smiles: SMILES строка соединения

        Returns:
            Dict: Предсказанные классы и вероятности
        """
        if not all([self.structural_model, self.activity_model, self.effect_model]):
            logger.error("Модели не обучены. Сначала обучите модели.")
            return {"error": "Модели не обучены"}

        try:
            # Извлечение фич
            features = self.extract_features_from_smiles(smiles)
            if not features:
                return {"error": "Не удалось извлечь фичи из SMILES"}

            X = np.array([list(features.values())])

            # Предсказания
            predictions = {}

            # Структурные классы
            X_struct = self.scalers["structural"].transform(X)
            struct_pred_binary = self.structural_model.predict(X_struct)[0]

            # Получение названий классов
            if hasattr(self.multilabel_binarizers["structural"], "classes_"):
                struct_classes = []
                struct_probs = []

                # Для каждой модели в MultiOutputClassifier получаем вероятность
                for i, estimator in enumerate(self.structural_model.estimators_):
                    if struct_pred_binary[i] == 1:
                        class_name = self.multilabel_binarizers["structural"].classes_[
                            i
                        ]
                        # Получаем вероятность предсказания
                        if hasattr(estimator, "predict_proba"):
                            prob = estimator.predict_proba(X_struct)[0][1]
                            struct_probs.append(prob)
                        else:
                            struct_probs.append(0.5)
                        struct_classes.append(class_name)

                predictions["structural_classes"] = {
                    "classes": struct_classes,
                    "probabilities": struct_probs,
                    "confidence": np.mean(struct_probs) if struct_probs else 0,
                }

            # Механизмы (фармакологическая активность)
            X_mech = self.scalers["mechanism"].transform(X)
            mech_pred_binary = self.activity_model.predict(X_mech)[0]

            if hasattr(self.multilabel_binarizers["mechanism"], "classes_"):
                mech_classes = []
                mech_probs = []

                for i, estimator in enumerate(self.activity_model.estimators_):
                    if mech_pred_binary[i] == 1:
                        class_name = self.multilabel_binarizers["mechanism"].classes_[i]
                        if hasattr(estimator, "predict_proba"):
                            prob = estimator.predict_proba(X_mech)[0][1]
                            mech_probs.append(prob)
                        else:
                            mech_probs.append(0.5)
                        mech_classes.append(class_name)

                predictions["mechanisms"] = {
                    "classes": mech_classes,
                    "probabilities": mech_probs,
                    "confidence": np.mean(mech_probs) if mech_probs else 0,
                }

                # Определение группы фармакологической активности
                activity_group = self._determine_pharmacological_group(mech_classes)
                predictions["pharmacological_activity_group"] = activity_group

            # Эффекты
            X_effect = self.scalers["effect"].transform(X)
            effect_pred_binary = self.effect_model.predict(X_effect)[0]

            if hasattr(self.multilabel_binarizers["effect"], "classes_"):
                effect_classes = []
                effect_probs = []

                for i, estimator in enumerate(self.effect_model.estimators_):
                    if effect_pred_binary[i] == 1:
                        class_name = self.multilabel_binarizers["effect"].classes_[i]
                        if hasattr(estimator, "predict_proba"):
                            prob = estimator.predict_proba(X_effect)[0][1]
                            effect_probs.append(prob)
                        else:
                            effect_probs.append(0.5)
                        effect_classes.append(class_name)

                predictions["biological_effects"] = {
                    "classes": effect_classes,
                    "probabilities": effect_probs,
                    "confidence": np.mean(effect_probs) if effect_probs else 0,
                }

            # Химические свойства
            predictions["chemical_properties"] = {
                "molecular_weight": features.get("mol_weight", 0),
                "logp": features.get("logp", 0),
                "tpsa": features.get("tpsa", 0),
                "num_h_donors": features.get("num_h_donors", 0),
                "num_h_acceptors": features.get("num_h_acceptors", 0),
                "num_rotatable_bonds": features.get("num_rotatable_bonds", 0),
                "num_rings": features.get("num_aromatic_rings", 0)
                + features.get("num_aliphatic_rings", 0),
            }

            # Рекомендации по применению
            recommendations = self._generate_recommendations(predictions)
            predictions["recommendations"] = recommendations

            # Увеличение счетчика классифицированных
            self.stats["total_classified"] += 1

            return predictions

        except Exception as e:
            logger.error(f"Ошибка предсказания для SMILES {smiles}: {e}")
            return {"error": str(e)}

    def _determine_pharmacological_group(self, mechanisms: List[str]) -> str:
        """
        Определение группы фармакологической активности на основе механизмов.

        Args:
            mechanisms: Список механизмов действия

        Returns:
            str: Группа фармакологической активности
        """
        if not mechanisms:
            return "Неизвестно"

        scores = defaultdict(int)

        for group, keywords in self.PHARMACOLOGICAL_ACTIVITY_GROUPS.items():
            for mechanism in mechanisms:
                for keyword in keywords:
                    if keyword.lower() in mechanism.lower():
                        scores[group] += 1

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return "Другая активность"

    def _generate_recommendations(self, predictions: Dict) -> List[str]:
        """
        Генерация рекомендаций на основе предсказаний.

        Args:
            predictions: Предсказания классификатора

        Returns:
            List: Список рекомендаций
        """
        recommendations = []

        # Рекомендации на основе структурного класса
        struct_classes = predictions.get("structural_classes", {}).get("classes", [])
        if struct_classes:
            recommendations.append(
                f"Относится к структурным классам: {', '.join(struct_classes[:3])}"
            )

        # Рекомендации на основе активности
        activity_group = predictions.get("pharmacological_activity_group")
        if activity_group and activity_group != "Неизвестно":
            recommendations.append(
                f"Основная фармакологическая группа: {activity_group}"
            )

        # Рекомендации на основе эффектов
        effects = predictions.get("biological_effects", {}).get("classes", [])
        if effects:
            recommendations.append(
                f"Ожидаемые биологические эффекты: {', '.join(effects[:3])}"
            )

        # Рекомендации на основе свойств
        props = predictions.get("chemical_properties", {})
        if props.get("logp", 0) > 5:
            recommendations.append(
                "Высокий LogP указывает на хорошую липофильность и проникновение через мембраны"
            )
        if props.get("num_h_donors", 0) > 5:
            recommendations.append(
                "Большое количество доноров водорода может улучшить растворимость"
            )
        if props.get("tpsa", 0) < 140:
            recommendations.append("Низкая TPSA способствует хорошей биодоступности")

        return recommendations

    def classify_all_database_compounds(
        self, batch_size: int = 50, update_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Классификация всех соединений в базе данных.

        Args:
            batch_size: Размер батча для обработки
            update_existing: Обновлять ли существующие записи

        Returns:
            Dict: Статистика классификации
        """
        if not MODELS_AVAILABLE:
            logger.error("Модели Django не доступны для работы с БД")
            return {"error": "Модели Django не доступны"}

        try:
            # Получение соединений для классификации
            if update_existing:
                compounds = Chronobiotic.objects.all()
            else:
                # Только соединения без классификации
                compounds = Chronobiotic.objects.filter(classf__isnull=True).distinct()

            total_compounds = compounds.count()

            logger.info(f"Начинаю классификацию {total_compounds} соединений")

            stats = {
                "total": total_compounds,
                "classified": 0,
                "failed": 0,
                "structural_classes": defaultdict(int),
                "activity_groups": defaultdict(int),
                "effects": defaultdict(int),
                "confidence_distribution": {
                    "high": 0,  # > 0.8
                    "medium": 0,  # 0.5-0.8
                    "low": 0,  # < 0.5
                },
            }

            # Классификация по батчам
            for i in range(0, total_compounds, batch_size):
                batch = compounds[i : i + batch_size]

                for compound in batch:
                    try:
                        smiles = compound.smiles
                        if not smiles:
                            logger.warning(f"Соединение {compound.id} не имеет SMILES")
                            stats["failed"] += 1
                            continue

                        # Предсказание классов
                        predictions = self.predict_compound_classes(smiles)

                        if "error" in predictions:
                            logger.warning(
                                f"Ошибка классификации соединения {compound.id}: {predictions['error']}"
                            )
                            stats["failed"] += 1
                            continue

                        # Обновление статистики
                        stats["classified"] += 1

                        # Структурные классы
                        struct_classes = predictions.get("structural_classes", {}).get(
                            "classes", []
                        )
                        for cls in struct_classes:
                            stats["structural_classes"][cls] += 1

                        # Группы активности
                        activity_group = predictions.get(
                            "pharmacological_activity_group", "Неизвестно"
                        )
                        stats["activity_groups"][activity_group] += 1

                        # Эффекты
                        effects = predictions.get("biological_effects", {}).get(
                            "classes", []
                        )
                        for effect in effects:
                            stats["effects"][effect] += 1

                        # Распределение уверенности
                        conf = predictions.get("structural_classes", {}).get(
                            "confidence", 0
                        )
                        if conf > 0.8:
                            stats["confidence_distribution"]["high"] += 1
                        elif conf > 0.5:
                            stats["confidence_distribution"]["medium"] += 1
                        else:
                            stats["confidence_distribution"]["low"] += 1

                        # Обновление базы данных
                        self._update_compound_in_database(compound, predictions)

                    except Exception as e:
                        logger.error(
                            f"Ошибка классификации соединения {compound.id}: {e}"
                        )
                        stats["failed"] += 1

                logger.info(
                    f"Обработано {min(i + batch_size, total_compounds)} из {total_compounds} соединений"
                )

            # Обновление общей статистики
            self.stats["last_database_update"] = datetime.now().isoformat()
            self.stats["database_stats"] = stats

            # Сохранение обновленной статистики
            self.save_models()

            logger.info(
                f"Классификация завершена. Успешно: {stats['classified']}, Неудачно: {stats['failed']}"
            )

            return {
                "status": "success",
                "statistics": stats,
                "summary": {
                    "success_rate": (
                        (stats["classified"] / total_compounds * 100)
                        if total_compounds > 0
                        else 0
                    ),
                    "most_common_structural_class": max(
                        stats["structural_classes"].items(),
                        key=lambda x: x[1],
                        default=("Нет данных", 0),
                    )[0],
                    "most_common_activity_group": max(
                        stats["activity_groups"].items(),
                        key=lambda x: x[1],
                        default=("Нет данных", 0),
                    )[0],
                },
            }

        except Exception as e:
            logger.error(f"Ошибка классификации соединений БД: {e}")
            return {"status": "error", "message": str(e)}

    def _update_compound_in_database(
        self, compound: Chronobiotic, predictions: Dict
    ) -> None:
        """
        Обновление соединения в базе данных на основе предсказаний.

        Args:
            compound: Объект соединения
            predictions: Предсказания классификатора
        """
        try:
            with transaction.atomic():
                # Структурные классы
                struct_classes = predictions.get("structural_classes", {}).get(
                    "classes", []
                )
                for class_name in struct_classes:
                    bioclass, created = Bioclass.objects.get_or_create(
                        nameclass=class_name, defaults={"nameclass": class_name}
                    )
                    compound.classf.add(bioclass)

                # Механизмы (фармакологическая активность)
                mechanisms = predictions.get("mechanisms", {}).get("classes", [])
                for mechanism_name in mechanisms:
                    mechanism, created = Mechanism.objects.get_or_create(
                        mechanismname=mechanism_name,
                        defaults={"mechanismname": mechanism_name},
                    )
                    compound.mechanisms.add(mechanism)

                # Эффекты
                effects = predictions.get("biological_effects", {}).get("classes", [])
                for effect_name in effects:
                    effect, created = Effect.objects.get_or_create(
                        Effectname=effect_name, defaults={"Effectname": effect_name}
                    )
                    compound.effect.add(effect)

                # Сохранение полных предсказаний в поле description или отдельном поле
                if hasattr(compound, "description"):
                    # Добавляем информацию о классификации в описание
                    classification_info = f"\n\n[Автоматическая классификация]\n"
                    classification_info += (
                        f"Структурные классы: {', '.join(struct_classes[:3])}\n"
                    )
                    classification_info += f"Фармакологическая группа: {predictions.get('pharmacological_activity_group', 'Неизвестно')}\n"
                    classification_info += (
                        f"Биологические эффекты: {', '.join(effects[:3])}\n"
                    )
                    classification_info += f"Уверенность классификации: {predictions.get('structural_classes', {}).get('confidence', 0):.2f}"

                    if not compound.description:
                        compound.description = classification_info
                    else:
                        compound.description += classification_info

                compound.save()

        except Exception as e:
            logger.error(f"Ошибка обновления соединения {compound.id}: {e}")

    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по базе данных.

        Returns:
            Dict: Статистика базы данных
        """
        if not MODELS_AVAILABLE:
            return {"error": "Модели Django не доступны"}

        try:
            # Общая статистика
            total_compounds = Chronobiotic.objects.count()
            classified_compounds = (
                Chronobiotic.objects.filter(classf__isnull=False).distinct().count()
            )

            # Статистика по классам
            bioclass_stats = list(
                Bioclass.objects.annotate(count=models.Count("chronobiotic"))
                .values("nameclass", "count")
                .order_by("-count")[:10]
            )

            # Статистика по механизмам
            mechanism_stats = list(
                Mechanism.objects.annotate(count=models.Count("chronobiotic"))
                .values("mechanismname", "count")
                .order_by("-count")[:10]
            )

            # Статистика по эффектам
            effect_stats = list(
                Effect.objects.annotate(count=models.Count("chronobiotic"))
                .values("Effectname", "count")
                .order_by("-count")[:10]
            )

            return {
                "total_compounds": total_compounds,
                "classified_compounds": classified_compounds,
                "classification_coverage": (
                    (classified_compounds / total_compounds * 100)
                    if total_compounds > 0
                    else 0
                ),
                "top_structural_classes": bioclass_stats,
                "top_mechanisms": mechanism_stats,
                "top_effects": effect_stats,
                "classifier_statistics": self.stats,
            }

        except Exception as e:
            logger.error(f"Ошибка получения статистики БД: {e}")
            return {"error": str(e)}

    def export_classification_report(
        self, output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Экспорт отчета о классификации.

        Args:
            output_format: Формат отчета ('json', 'csv')

        Returns:
            Dict: Результат экспорта
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Подготовка данных отчета
            report = {
                "timestamp": datetime.now().isoformat(),
                "classifier_statistics": self.stats,
                "model_information": {
                    "structural_model": (
                        str(type(self.structural_model).__name__)
                        if self.structural_model
                        else "Не обучена"
                    ),
                    "activity_model": (
                        str(type(self.activity_model).__name__)
                        if self.activity_model
                        else "Не обучена"
                    ),
                    "effect_model": (
                        str(type(self.effect_model).__name__)
                        if self.effect_model
                        else "Не обучена"
                    ),
                    "feature_count": (
                        len(self.feature_names) if self.feature_names else 0
                    ),
                    "available_classes": self.get_existing_classes_from_db(),
                },
                "database_statistics": self.get_database_statistics(),
            }

            if output_format == "json":
                output_path = os.path.join(
                    self.model_dir, f"classification_report_{timestamp}.json"
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

                return {"status": "success", "file_path": output_path, "format": "json"}

            elif output_format == "csv":
                # Экспорт в CSV
                output_path = os.path.join(
                    self.model_dir, f"classification_report_{timestamp}.csv"
                )

                # Подготовка данных для CSV
                csv_data = []

                # Добавляем статистику
                for key, value in self.stats.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            csv_data.append([f"{key}.{subkey}", subvalue])
                    else:
                        csv_data.append([key, value])

                # Сохранение CSV
                import csv

                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Parameter", "Value"])
                    writer.writerows(csv_data)

                return {"status": "success", "file_path": output_path, "format": "csv"}

            else:
                return {
                    "status": "error",
                    "message": f"Неподдерживаемый формат: {output_format}",
                }

        except Exception as e:
            logger.error(f"Ошибка экспорта отчета: {e}")
            return {"status": "error", "message": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """
        Получение статуса классификатора.

        Returns:
            Dict: Статус и информация о классификаторе
        """
        models_loaded = {
            "structural": self.structural_model is not None,
            "activity": self.activity_model is not None,
            "effect": self.effect_model is not None,
        }

        db_stats = self.get_database_statistics()

        return {
            "status": "ready" if all(models_loaded.values()) else "not_ready",
            "models_loaded": models_loaded,
            "database_statistics": db_stats,
            "classifier_statistics": {
                "total_classified": self.stats.get("total_classified", 0),
                "last_training": self.stats.get("last_training"),
                "last_database_update": self.stats.get("last_database_update"),
                "training_samples": self.stats.get("training_samples", 0),
            },
            "available_features": len(self.feature_names),
            "model_directory": self.model_dir,
        }


# Фабричная функция для получения классификатора
def get_chemical_classifier() -> ChemicalClassifier:
    """
    Получение экземпляра химического классификатора.

    Returns:
        ChemicalClassifier: Экземпляр классификатора
    """
    return ChemicalClassifier()


# Функции для интеграции с Django
def train_classifier_task() -> Dict:
    """
    Задача для обучения классификатора.

    Returns:
        Dict: Результат обучения
    """
    classifier = get_chemical_classifier()
    return classifier.train_models_from_database()


def classify_database_task() -> Dict:
    """
    Задача для классификации всех соединений в БД.

    Returns:
        Dict: Результат классификации
    """
    classifier = get_chemical_classifier()
    return classifier.classify_all_database_compounds()


def classify_single_compound_task(smiles: str) -> Dict:
    """
    Задача для классификации одного соединения.

    Args:
        smiles: SMILES строка

    Returns:
        Dict: Результат классификации
    """
    classifier = get_chemical_classifier()
    return classifier.predict_compound_classes(smiles)


def get_classifier_status_task() -> Dict:
    """
    Задача для получения статуса классификатора.

    Returns:
        Dict: Статус классификатора
    """
    classifier = get_chemical_classifier()
    return classifier.get_status()


if __name__ == "__main__":
    # Тестирование классификатора
    import sys

    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Инициализация Django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chronobiotic.settings")
    import django

    django.setup()

    classifier = ChemicalClassifier()

    print("=" * 80)
    print("Химический классификатор ChronobioticsDB")
    print("=" * 80)

    # Получение статуса
    status = classifier.get_status()
    print(f"Статус: {status['status']}")
    print(f"Модели загружены: {status['models_loaded']}")
    print(
        f"База данных: {status['database_statistics'].get('total_compounds', 0)} соединений"
    )

    # Пример классификации
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Аспирин
    print(f"\nТестовая классификация для SMILES: {test_smiles}")

    result = classifier.predict_compound_classes(test_smiles)
    if "error" not in result:
        print(
            f"Структурные классы: {result.get('structural_classes', {}).get('classes', [])}"
        )
        print(
            f"Фармакологическая группа: {result.get('pharmacological_activity_group', 'Неизвестно')}"
        )
        print(
            f"Биологические эффекты: {result.get('biological_effects', {}).get('classes', [])}"
        )
        print(f"Рекомендации: {result.get('recommendations', [])}")
    else:
        print(f"Ошибка: {result['error']}")

    print("\n" + "=" * 80)
