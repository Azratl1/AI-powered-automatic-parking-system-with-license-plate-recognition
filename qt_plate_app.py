from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDialog,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "PySide6 не установлен. Установите: .\\.venv\\Scripts\\python.exe -m pip install PySide6"
    ) from exc

from demo_my_images import choose_best_ocr_for_detection
from full_plate_pipeline import YoloPlateDetector, draw_bbox, load_ocr_model
from plate_storage import PlateStorage, normalize_plate


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
OPERATOR_LOGIN_PREFIX = "КПП-"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qt-приложение СКУД для автотранспорта.")
    parser.add_argument("--ocr-weights", default="ocr_checkpoints/ocr_best.pt")
    parser.add_argument("--yolo-weights", default="runs/detect/runs_yolo/plate_detector_ru2/weights/best.pt")
    parser.add_argument("--pg-dsn", default=None, help="DSN PostgreSQL. По умолчанию берется PLATE_DB_DSN.")
    parser.add_argument("--operator", default="", help="Логин оператора.")
    parser.add_argument("--images-dir", default="my_images", help="Папка с последними снимками.")
    return parser.parse_args()


def cv_to_qpixmap(image: np.ndarray) -> QPixmap:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    q_img = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_img.copy())


class ImageBox(QLabel):
    def __init__(self, title: str, width: int, height: int) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self.setText(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(width, height)
        self.setStyleSheet(
            "QLabel{border:1px solid #2f3647;border-radius:8px;background:#151925;color:#8f9bb3;}"
        )

    def set_cv_image(self, image: np.ndarray) -> None:
        self._pixmap = cv_to_qpixmap(image)
        self._refresh_pixmap()

    def clear_image(self, title: str) -> None:
        self._pixmap = None
        self.clear()
        self.setText(title)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class OperatorLoginDialog(QDialog):
    def __init__(self, default_login: str = "") -> None:
        super().__init__()
        self.setWindowTitle("Авторизация сотрудника")
        self.resize(420, 245)
        self.setObjectName("loginDialog")
        self.mode: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(12)

        title = QLabel("Вход в СКУД")
        title.setObjectName("loginTitle")
        layout.addWidget(title)

        form_box = QGroupBox("Сотрудник")
        form = QFormLayout()
        form.setContentsMargins(14, 18, 14, 14)
        form.setSpacing(10)
        default_number = default_login.strip()
        if default_number.startswith(OPERATOR_LOGIN_PREFIX):
            default_number = default_number[len(OPERATOR_LOGIN_PREFIX):]
        self.login_prefix = QLabel(OPERATOR_LOGIN_PREFIX)
        self.login_prefix.setObjectName("loginPrefix")
        self.login_number = QLineEdit(default_number)
        self.login_number.setPlaceholderText("например, 12")
        self.full_name = QLineEdit()
        self.full_name.setPlaceholderText("только при регистрации")
        login_row = QHBoxLayout()
        login_row.setSpacing(8)
        login_row.addWidget(self.login_prefix)
        login_row.addWidget(self.login_number, stretch=1)
        form.addRow("Номер КПП", login_row)
        form.addRow("ФИО", self.full_name)
        form_box.setLayout(form)
        layout.addWidget(form_box)

        buttons = QHBoxLayout()
        self.cancel_button = QPushButton("Отмена")
        self.register_button = QPushButton("Регистрация")
        self.enter_button = QPushButton("Вход")
        self.enter_button.setObjectName("loginButton")
        self.cancel_button.clicked.connect(self.reject)  # type: ignore[arg-type]
        self.register_button.clicked.connect(self._register_if_valid)  # type: ignore[arg-type]
        self.enter_button.clicked.connect(self._login_if_valid)  # type: ignore[arg-type]
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.register_button)
        buttons.addWidget(self.enter_button)
        layout.addLayout(buttons)

        self.login_number.setFocus()
        self.login_number.returnPressed.connect(self._login_if_valid)  # type: ignore[arg-type]
        self.full_name.returnPressed.connect(self._register_if_valid)  # type: ignore[arg-type]
        self.setStyleSheet(
            """
            QDialog#loginDialog { background: #0f131d; color: #e8ecf3; }
            QLabel { color: #dce5f5; font-size: 14px; }
            QLabel#loginTitle { color: #ffffff; font-size: 24px; font-weight: 800; }
            QLabel#loginPrefix {
                background: #1b2130; border: 1px solid #334057; border-radius: 7px;
                color: #ffffff; font-weight: 800; padding: 8px 12px;
            }
            QGroupBox {
                border: 1px solid #2f3647; border-radius: 10px; margin-top: 12px;
                background: #101622; font-weight: 700; color: #dce5f5;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #dce5f5;
            }
            QLineEdit {
                background: #151925; border: 1px solid #2f3647; border-radius: 7px;
                color: #e8ecf3; padding: 8px;
            }
            QLineEdit:focus { border-color: #2b91bd; }
            QPushButton {
                background: #273247; color: #e8ecf3; border: none; border-radius: 8px;
                padding: 10px 18px; font-weight: 700;
            }
            QPushButton:hover { background: #33415c; }
            QPushButton#loginButton { background: #187a4d; color: #ffffff; }
            QPushButton#loginButton:hover { background: #1f9360; }
            """
        )

    def _validate_number(self) -> bool:
        number = self.login_number.text().strip()
        if not number:
            QMessageBox.warning(self, "Авторизация", "Введите номер сотрудника КПП.")
            return False
        if not number.isdigit():
            QMessageBox.warning(self, "Авторизация", "Номер КПП должен состоять только из цифр.")
            return False
        return True

    def _login_if_valid(self) -> None:
        if not self._validate_number():
            return
        self.mode = "login"
        self.accept()

    def _register_if_valid(self) -> None:
        if not self._validate_number():
            return
        if not self.full_name.text().strip():
            QMessageBox.warning(self, "Регистрация", "Для первичной регистрации введите ФИО.")
            return
        self.mode = "register"
        self.accept()

    def login_value(self) -> str:
        return f"{OPERATOR_LOGIN_PREFIX}{self.login_number.text().strip()}"


class MainWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

        self.current_image_path: str | None = None
        self.current_image: np.ndarray | None = None
        self.pending_event: dict[str, object] | None = None
        self.last_seen_image_signature: tuple[str, float] | None = None
        self.is_recognizing = False
        self.operator_login = args.operator
        self.operator_full_name = ""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_detector: YoloPlateDetector | None = None
        self.ocr_model = None
        self.converter = None
        self.storage = PlateStorage(args.pg_dsn)

        self._authenticate_operator()
        self.setWindowTitle("СКУД автотранспорта")
        self.setFixedSize(1800, 940)
        self._setup_ui()
        self._apply_dark_theme()
        self._load_catalogs()
        self._load_models()
        self.load_latest_image()
        if self.current_image is not None:
            self.run_recognition()
        self._start_image_watch()
        self.refresh_parking()
        self.refresh_audit()

    def _authenticate_operator(self) -> None:
        while True:
            dialog = OperatorLoginDialog(self.operator_login)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                raise SystemExit("Авторизация отменена.")

            login = dialog.login_value()
            if dialog.mode == "register":
                operator = self.storage.authenticate_operator(login, dialog.full_name.text())
                self.operator_login = operator["login"]
                self.operator_full_name = operator.get("full_name") or ""
                return

            operator = self.storage.get_operator(login)
            if operator is None:
                QMessageBox.warning(
                    self,
                    "Авторизация",
                    "Сотрудник не найден. Сначала выполните первичную регистрацию.",
                )
                continue
            self.operator_login = operator["login"]
            self.operator_full_name = operator.get("full_name") or ""
            return

    def _setup_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        self.main_tabs = QTabWidget()
        root_layout.addWidget(self.main_tabs)

        work_tab = QWidget()
        work_layout = QVBoxLayout(work_tab)
        work_layout.setContentsMargins(0, 0, 0, 0)
        work_layout.setSpacing(12)

        controls = QHBoxLayout()
        self.btn_open = QPushButton("Выбрать фото")
        self.btn_latest = QPushButton("Последнее из my_images")
        self.btn_run = QPushButton("Распознать")
        self.btn_manual_entry = QPushButton("Ручной въезд")
        self.btn_refresh_parking = QPushButton("Обновить парковку")
        self.btn_refresh_audit = QPushButton("Обновить аудит")
        self.btn_db_counts = QPushButton("Проверить БД")
        self.btn_open.clicked.connect(self.open_image)  # type: ignore[arg-type]
        self.btn_latest.clicked.connect(self.load_latest_and_recognize)  # type: ignore[arg-type]
        self.btn_run.clicked.connect(self.run_recognition)  # type: ignore[arg-type]
        self.btn_manual_entry.clicked.connect(self.start_manual_entry)  # type: ignore[arg-type]
        self.btn_refresh_parking.clicked.connect(self.refresh_parking)  # type: ignore[arg-type]
        self.btn_refresh_audit.clicked.connect(self.refresh_audit)  # type: ignore[arg-type]
        self.btn_db_counts.clicked.connect(self.show_db_counts)  # type: ignore[arg-type]
        self.btn_open.hide()
        self.btn_latest.hide()
        self.btn_run.hide()
        controls.addWidget(self.btn_manual_entry)
        controls.addStretch(1)
        controls.addWidget(self.btn_refresh_parking)
        controls.addWidget(self.btn_db_counts)
        work_layout.addLayout(controls)

        body = QHBoxLayout()
        body.setSpacing(12)
        work_layout.addLayout(body, stretch=1)

        left_panel = QWidget()
        left_panel.setFixedWidth(880)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        self.main_image_box = ImageBox("Изображение автомобиля", 880, 720)
        left.addWidget(self.main_image_box)
        body.addWidget(left_panel)

        middle_panel = QWidget()
        middle_panel.setFixedWidth(430)
        middle = QVBoxLayout(middle_panel)
        middle.setContentsMargins(0, 0, 0, 0)
        self.crop_box = ImageBox("Фрагмент номера", 430, 250)
        self.plate_box = ImageBox("Вид для OCR", 430, 210)
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setFixedSize(430, 250)
        self.info_box.setPlainText("Ожидание изображения.")
        middle.addWidget(self.crop_box)
        middle.addWidget(self.plate_box)
        middle.addWidget(self.info_box)
        body.addWidget(middle_panel)

        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(430)
        self.tabs.addTab(self._build_access_tab(), "Доступ")
        body.addWidget(self.tabs)
        self.main_tabs.addTab(work_tab, "СКУД")
        self.main_tabs.addTab(self._build_parking_tab(), "На парковке")
        self.main_tabs.addTab(self._build_audit_tab(), "Аудит")
        self.statusBar().showMessage("Готово")

    def _build_access_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.status_label = QLabel("Нет распознанного номера")
        self.status_label.setObjectName("decisionStatus")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form_box = QGroupBox("Карточка доступа")
        form = QFormLayout(form_box)
        self.entry_at = QLineEdit()
        self.entry_at.setReadOnly(True)
        self.detected_plate = QLineEdit()
        self.raw_text = QLineEdit()
        self.raw_text.setReadOnly(True)
        self.operator = QLineEdit(self._operator_display_name())
        self.operator.setReadOnly(True)
        self.owner_name = QLineEdit()
        self.owner_phone = QLineEdit()
        self.visitor_code = QLineEdit()
        self.visitor_code.setReadOnly(True)
        self.btn_generate_visitor_code = QPushButton("Сгенерировать")
        self.btn_generate_visitor_code.clicked.connect(self.generate_visitor_code)  # type: ignore[arg-type]
        visitor_code_row = QHBoxLayout()
        visitor_code_row.addWidget(self.visitor_code, stretch=1)
        visitor_code_row.addWidget(self.btn_generate_visitor_code)
        self.vehicle_make = QComboBox()
        self.vehicle_make.setEditable(True)
        self.vehicle_make.currentTextChanged.connect(self._reload_models_for_make)  # type: ignore[arg-type]
        self.vehicle_model = QComboBox()
        self.vehicle_model.setEditable(True)
        self.vehicle_color = QLineEdit()
        self.vehicle_year = QLineEdit()
        self.visitor_name = QLineEdit()
        self.visit_purpose = QLineEdit()
        self.comment = QLineEdit()
        self.note = QLineEdit()

        form.addRow("Дата заезда", self.entry_at)
        form.addRow("Номер", self.detected_plate)
        form.addRow("OCR без правки", self.raw_text)
        form.addRow("Оператор", self.operator)
        form.addRow("ФИО посетителя", self.owner_name)
        form.addRow("Телефон посетителя", self.owner_phone)
        form.addRow("Код посетителя", visitor_code_row)
        form.addRow("Марка", self.vehicle_make)
        form.addRow("Модель", self.vehicle_model)
        form.addRow("Цвет", self.vehicle_color)
        form.addRow("Год", self.vehicle_year)
        form.addRow("Цель визита", self.visit_purpose)
        form.addRow("Комментарий авто", self.comment)
        form.addRow("Комментарий решения", self.note)
        layout.addWidget(form_box)

        buttons = QHBoxLayout()
        self.btn_deny = QPushButton("Отказ")
        self.btn_allow = QPushButton("Разрешить")
        self.btn_deny.setObjectName("denyButton")
        self.btn_allow.setObjectName("allowButton")
        self.btn_deny.clicked.connect(lambda: self.save_decision("denied"))  # type: ignore[arg-type]
        self.btn_allow.clicked.connect(lambda: self.save_decision("allowed"))  # type: ignore[arg-type]
        buttons.addWidget(self.btn_deny)
        buttons.addWidget(self.btn_allow)
        layout.addLayout(buttons)
        self._set_decision_enabled(False)

        return tab

    def _build_parking_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        exit_buttons = QHBoxLayout()
        self.btn_exit_vehicle = QPushButton("Подтвердить выезд")
        self.btn_exit_vehicle.setObjectName("denyButton")
        self.btn_exit_vehicle.clicked.connect(self.confirm_vehicle_exit)  # type: ignore[arg-type]
        exit_buttons.addStretch(1)
        exit_buttons.addWidget(self.btn_exit_vehicle)
        layout.addLayout(exit_buttons)

        self.parking_table = QTableWidget(0, 9)
        self.parking_table.setHorizontalHeaderLabels(
            [
                "ID",
                "Дата въезда",
                "Код",
                "Номер",
                "Марка",
                "Модель",
                "Цвет",
                "Посетитель",
                "Цель",
            ]
        )
        self.parking_table.setColumnHidden(0, True)
        self.parking_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.parking_table)
        return tab

    def _build_audit_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top = QHBoxLayout()
        top.addStretch(1)
        self.btn_refresh_audit_tab = QPushButton("Обновить аудит")
        self.btn_refresh_audit_tab.clicked.connect(self.refresh_audit)  # type: ignore[arg-type]
        top.addWidget(self.btn_refresh_audit_tab)
        layout.addLayout(top)
        self.audit_table = QTableWidget(0, 9)
        self.audit_table.setHorizontalHeaderLabels(
            [
                "Въезд",
                "Выезд",
                "Время",
                "Код",
                "Номер",
                "Марка",
                "Модель",
                "Посетитель",
                "Оператор",
            ]
        )
        self.audit_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.audit_table)
        return tab

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QDialog { background: #0f131d; color: #e8ecf3; }
            QWidget { color: #e8ecf3; font-size: 14px; }
            QPushButton {
                background: #1f6f94; color: white; border: none; border-radius: 8px;
                padding: 10px 14px; font-weight: 600;
            }
            QPushButton:hover { background: #238ab0; }
            QPushButton:disabled { background: #2b3444; color: #77849a; }
            QPushButton#allowButton { background: #187a4d; }
            QPushButton#allowButton:hover { background: #1f9360; }
            QPushButton#denyButton { background: #9b2d35; }
            QPushButton#denyButton:hover { background: #bd3942; }
            QTextEdit, QTableWidget, QTabWidget::pane {
                background: #151925; border: 1px solid #2f3647; border-radius: 8px; color: #e8ecf3;
            }
            QTabBar::tab {
                background: #1b2130; color: #c8d4e8; padding: 9px 14px;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
            }
            QTabBar::tab:selected { background: #1f6f94; color: white; }
            QHeaderView::section {
                background: #1b2130; color: #b8c2d9; border: 0; padding: 6px; font-weight: 600;
            }
            QLabel#pathLabel { color: #a8b3ca; }
            QLabel#decisionStatus {
                background: #151925; border: 1px solid #2f3647; border-radius: 8px;
                color: #ffffff; font-size: 16px; font-weight: 700; padding: 12px;
            }
            QLineEdit, QComboBox {
                background: #151925; border: 1px solid #2f3647; border-radius: 7px;
                color: #e8ecf3; padding: 7px;
            }
            QComboBox QAbstractItemView {
                background: #111827;
                color: #f3f7ff;
                selection-background-color: #1f6f94;
                selection-color: #ffffff;
                border: 1px solid #3b4860;
                outline: 0;
                padding: 4px;
            }
            QComboBox::drop-down {
                background: #1b2130;
                border-left: 1px solid #2f3647;
                width: 26px;
                border-top-right-radius: 7px;
                border-bottom-right-radius: 7px;
            }
            QGroupBox {
                border: 1px solid #2f3647; border-radius: 8px; margin-top: 14px;
                background: #101622; font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #dce5f5;
            }
            QStatusBar { background: #121726; color: #9fb6d1; }
            """
        )

    def _load_catalogs(self) -> None:
        makes = self.storage.list_vehicle_makes()
        self.vehicle_make.blockSignals(True)
        self.vehicle_make.clear()
        self.vehicle_make.addItems(makes)
        self.vehicle_make.setCurrentText("")
        self.vehicle_make.blockSignals(False)
        self._reload_models_for_make("")

    def _reload_models_for_make(self, make_name: str) -> None:
        current = self.vehicle_model.currentText().strip()
        models = self.storage.list_vehicle_models(make_name)
        self.vehicle_model.blockSignals(True)
        self.vehicle_model.clear()
        self.vehicle_model.addItems(models)
        self.vehicle_model.setCurrentText(current)
        self.vehicle_model.blockSignals(False)

    def generate_visitor_code(self) -> None:
        try:
            self.visitor_code.setText(self.storage.generate_visitor_code())
            self.statusBar().showMessage("Код посетителя сгенерирован")
        except Exception as exc:
            QMessageBox.warning(self, "Код посетителя", str(exc))

    def _load_models(self) -> None:
        try:
            self.statusBar().showMessage(f"Загрузка моделей: {self.device}...")
            yolo_path = Path(self.args.yolo_weights)
            if not yolo_path.exists():
                raise FileNotFoundError(f"Веса YOLO не найдены: {yolo_path}")
            self.yolo_detector = YoloPlateDetector(str(yolo_path), device=str(self.device))
            self.ocr_model, self.converter = load_ocr_model(self.args.ocr_weights, self.device)
            self.statusBar().showMessage(f"Модели загружены. Устройство: {self.device}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка загрузки моделей", str(exc))
            raise

    def load_latest_image(self) -> None:
        latest = self._latest_image_path()
        if latest is None:
            return
        self._load_image(str(latest))

    def load_latest_and_recognize(self) -> None:
        self.load_latest_image()
        if self.current_image is not None:
            self.run_recognition()

    def _latest_image_path(self) -> Path | None:
        images_dir = Path(self.args.images_dir)
        if not images_dir.exists():
            self.statusBar().showMessage(f"Папка не найдена: {images_dir}")
            return None
        files = [
            path for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
        if not files:
            self.statusBar().showMessage(f"В папке нет изображений: {images_dir}")
            return None
        return max(files, key=lambda path: path.stat().st_mtime)

    def _start_image_watch(self) -> None:
        self.image_watch_timer = QTimer(self)
        self.image_watch_timer.setInterval(2000)
        self.image_watch_timer.timeout.connect(self.check_latest_image)  # type: ignore[arg-type]
        self.image_watch_timer.start()

    def check_latest_image(self) -> None:
        if self.is_recognizing:
            return
        latest = self._latest_image_path()
        if latest is None:
            return
        signature = (str(latest.resolve()), latest.stat().st_mtime)
        if signature == self.last_seen_image_signature:
            return
        self._load_image(str(latest))
        self.run_recognition()

    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать изображение",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if file_path:
            self._load_image(file_path)
            self.run_recognition()

    def _load_image(self, file_path: str) -> None:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "Ошибка", "Не удалось прочитать изображение.")
            return
        self.current_image_path = file_path
        self.current_image = image
        path_obj = Path(file_path)
        if path_obj.exists():
            self.last_seen_image_signature = (str(path_obj.resolve()), path_obj.stat().st_mtime)
        self.pending_event = None
        self.main_image_box.set_cv_image(image)
        self.crop_box.clear_image("Фрагмент номера")
        self.plate_box.clear_image("Вид для OCR")
        self.info_box.setPlainText("Ожидание распознавания.")
        self.status_label.setText("Ожидание распознавания")
        self._clear_decision_form(keep_operator=True)
        self._set_decision_enabled(False)
        self.statusBar().showMessage("Изображение загружено")

    def run_recognition(self) -> None:
        if self.is_recognizing:
            return
        if self.current_image is None or self.current_image_path is None:
            QMessageBox.information(self, "Нет изображения", "Сначала выберите изображение.")
            return
        if self.yolo_detector is None or self.ocr_model is None or self.converter is None:
            QMessageBox.critical(self, "Ошибка", "Модели не инициализированы.")
            return

        self.btn_run.setEnabled(False)
        self.is_recognizing = True
        self.statusBar().showMessage("Распознавание...")
        QApplication.processEvents()
        try:
            image = self.current_image.copy()
            detection = self.yolo_detector.detect(image)
            if detection is None:
                raise RuntimeError("YOLO не нашел номер на изображении.")
            best_ocr = choose_best_ocr_for_detection(
                image=image,
                bbox=detection.bbox,
                ocr_model=self.ocr_model,
                converter=self.converter,
                device=self.device,
            )
            if best_ocr is None:
                raise RuntimeError("Не удалось подготовить фрагмент для OCR.")
            ocr_result, crop, plate_view, vis_bbox = best_ocr
            plate = ocr_result.text or "-"
            raw = ocr_result.raw_text or "-"
            entry_time = datetime.now(timezone.utc)
            timestamp = entry_time.astimezone().strftime("%Y-%m-%d %H:%M")

            vis = image.copy()
            draw_bbox(
                vis,
                vis_bbox,
                f"{plate if plate != '-' else raw} | {detection.source} | {detection.score:.2f}",
            )
            self.main_image_box.set_cv_image(vis)
            self.crop_box.set_cv_image(crop)
            self.plate_box.set_cv_image(plate_view)
            self.info_box.setPlainText(
                "\n".join(
                    [
                        f"Дата заезда: {timestamp}",
                        f"Номер: {plate}",
                    ]
                )
            )

            self.pending_event = {
                "detected_plate_text": plate,
                "raw_text": raw,
                "detector_source": detection.source,
                "detector_score": detection.score,
                "ocr_confidence": ocr_result.confidence,
                "bbox": vis_bbox,
                "image_path": self.current_image_path,
                "vis_image": vis,
                "crop_image": crop,
                "plate_image": plate_view,
                "entry_at": entry_time,
                "timestamp": timestamp,
            }
            self._fill_decision_form(
                plate=plate,
                raw=raw,
                entry_time=entry_time.astimezone(),
                context=self.storage.build_review_context(plate),
            )
            self._set_decision_enabled(True)
            self.tabs.setCurrentIndex(0)
            self.statusBar().showMessage("Номер распознан. Проверьте карточку доступа.")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка распознавания", str(exc))
            self.statusBar().showMessage("Ошибка распознавания")
        finally:
            self.is_recognizing = False
            self.btn_run.setEnabled(True)

    def start_manual_entry(self) -> None:
        now = datetime.now(timezone.utc)
        timestamp = now.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        self.pending_event = {
            "detected_plate_text": "",
            "raw_text": "",
            "detector_source": "manual",
            "detector_score": 0.0,
            "ocr_confidence": 0.0,
            "bbox": (0, 0, 0, 0),
            "image_path": self.current_image_path or "",
            "entry_at": now,
            "timestamp": timestamp,
        }
        self.status_label.setText("Ручной въезд\nЗаполните номер и данные посетителя.")
        self.entry_at.setText(timestamp)
        self.detected_plate.clear()
        self.raw_text.setText("ручной ввод")
        self.info_box.setPlainText(
            "\n".join(
                [
                    f"Дата заезда: {timestamp}",
                    "Номер: ручной ввод",
                ]
            )
        )
        self.operator.setText(self._operator_display_name())
        self.owner_name.clear()
        self.owner_phone.clear()
        self.visitor_code.clear()
        self.vehicle_make.setCurrentText("")
        self.vehicle_model.setCurrentText("")
        self.vehicle_color.clear()
        self.vehicle_year.clear()
        self.comment.clear()
        self.visitor_name.clear()
        self.visit_purpose.clear()
        self.note.clear()
        self._set_decision_enabled(True)
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage("Ручной въезд: заполните карточку и сохраните решение.")

    def _set_decision_enabled(self, enabled: bool) -> None:
        self.btn_allow.setEnabled(enabled)
        self.btn_deny.setEnabled(enabled)

    def _clear_decision_form(self, keep_operator: bool = False) -> None:
        operator = self.operator.text() if keep_operator else self._operator_display_name()
        for field in (
            self.entry_at,
            self.detected_plate,
            self.raw_text,
            self.owner_name,
            self.owner_phone,
            self.visitor_code,
            self.vehicle_color,
            self.vehicle_year,
            self.visitor_name,
            self.visit_purpose,
            self.comment,
            self.note,
        ):
            field.clear()
        self.vehicle_make.setCurrentText("")
        self.vehicle_model.setCurrentText("")
        self.operator.setText(operator)

    def _fill_decision_form(
        self,
        *,
        plate: str,
        raw: str,
        entry_time: datetime,
        context: dict,
    ) -> None:
        vehicle = context.get("vehicle") or {}
        visitor = context.get("visitor") or {}
        last_entry = context.get("last_entry")
        active_rule = context.get("active_rule")
        status = "Номер есть в базе" if vehicle else "Новый номер"
        last_text = "Предыдущих заездов нет"
        if last_entry:
            last_text = f"Последний заезд: {last_entry['entry_at']} | {self._decision_to_ru(last_entry['decision'])}"
        rule_text = ""
        if active_rule:
            rule_name = "запрет" if active_rule["rule_type"] == "deny" else "разрешение"
            rule_text = f"\nАктивное правило: {rule_name}"

        self.status_label.setText(f"{status}\n{last_text}{rule_text}")
        self.entry_at.setText(entry_time.strftime("%Y-%m-%d %H:%M"))
        self.detected_plate.setText(plate)
        self.raw_text.setText(raw)
        self.operator.setText(self._operator_display_name())
        self.owner_name.setText(visitor.get("visitor_name") or "")
        self.owner_phone.setText(visitor.get("visitor_phone") or "")
        self.visitor_code.setText(visitor.get("visitor_code") or "")
        self.vehicle_make.setCurrentText(vehicle.get("vehicle_make") or "")
        self._reload_models_for_make(self.vehicle_make.currentText())
        self.vehicle_model.setCurrentText(vehicle.get("vehicle_model") or "")
        self.vehicle_color.setText(vehicle.get("color") or "")
        self.vehicle_year.setText(str(vehicle.get("manufacture_year") or ""))
        self.comment.setText(vehicle.get("comment") or "")
        self.visitor_name.clear()
        self.visit_purpose.setText(visitor.get("visit_purpose") or "")
        self.note.clear()

    def _decision_values(self) -> dict[str, object]:
        year_text = self.vehicle_year.text().strip()
        return {
            "plate_text": normalize_plate(self.detected_plate.text()),
            "operator_name": self.operator_login,
            "owner_name": "",
            "owner_phone": "",
            "visitor_name": self.owner_name.text().strip(),
            "visitor_phone": self.owner_phone.text().strip(),
            "visitor_code": self.visitor_code.text().strip(),
            "vehicle_make": self.vehicle_make.currentText().strip(),
            "vehicle_model": self.vehicle_model.currentText().strip(),
            "color": self.vehicle_color.text().strip(),
            "manufacture_year": int(year_text) if year_text.isdigit() else None,
            "visit_purpose": self.visit_purpose.text().strip(),
            "comment": self.comment.text().strip(),
            "note": self.note.text().strip(),
        }

    def save_decision(self, decision: str) -> None:
        if self.pending_event is None:
            QMessageBox.information(self, "Нет события", "Сначала распознайте номер.")
            return
        values = self._decision_values()
        if decision == "allowed" and not values["visitor_code"]:
            values["visitor_code"] = self.storage.generate_visitor_code()
            self.visitor_code.setText(str(values["visitor_code"]))
        final_plate = values.pop("plate_text")
        if not final_plate:
            QMessageBox.warning(self, "Номер", "Номер не может быть пустым.")
            return
        try:
            artifact_paths = self._save_event_artifacts(str(final_plate), decision)
            event = self.storage.record_operator_decision(
                detected_plate_text=str(self.pending_event["detected_plate_text"]),
                plate_text=str(final_plate),
                raw_text=str(self.pending_event["raw_text"]),
                decision=decision,
                detector_source=str(self.pending_event["detector_source"]),
                detector_score=float(self.pending_event["detector_score"]),
                ocr_confidence=float(self.pending_event["ocr_confidence"]),
                bbox=self.pending_event["bbox"],  # type: ignore[arg-type]
                image_path=str(self.pending_event["image_path"]),
                vis_path=artifact_paths.get("vis_path"),
                crop_path=artifact_paths.get("crop_path"),
                plate_image_path=artifact_paths.get("plate_image_path"),
                entry_at=self.pending_event["entry_at"],  # type: ignore[arg-type]
                **values,
            )
            self._append_table_row(
                str(self.pending_event["timestamp"]),
                event["plate_text"],
                event["decision"],
                str(self.pending_event["detector_source"]),
                float(self.pending_event["ocr_confidence"]),
            )
            self.pending_event = None
            self._set_decision_enabled(False)
            self._load_catalogs()
            self.refresh_parking()
            self.refresh_audit()
            self.show_db_counts(silent=True)
            self.status_label.setText(
                f"Решение сохранено\nНомер: {event['plate_text']} | {self._decision_to_ru(event['decision'])}"
            )
            self.statusBar().showMessage("Решение сохранено в PostgreSQL")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))
            self.statusBar().showMessage("Ошибка сохранения решения")

    def _save_event_artifacts(self, plate: str, decision: str) -> dict[str, str]:
        if self.pending_event is None:
            return {}
        output_dir = Path("app_data") / "qt_events"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_plate = "".join(ch for ch in plate if ch.isalnum()) or "plate"
        prefix = f"{stamp}_{safe_plate}_{decision}"
        paths: dict[str, str] = {}

        artifacts = [
            ("vis_path", "car", self.pending_event.get("vis_image")),
            ("crop_path", "crop", self.pending_event.get("crop_image")),
            ("plate_image_path", "ocr", self.pending_event.get("plate_image")),
        ]
        for key, name, image in artifacts:
            if image is None:
                continue
            path = output_dir / f"{prefix}_{name}.jpg"
            if cv2.imwrite(str(path), image):  # type: ignore[arg-type]
                paths[key] = str(path)
        return paths

    def show_db_counts(self, silent: bool = False) -> None:
        try:
            counts = self.storage.get_table_counts()
            text = ", ".join(f"{name}: {count}" for name, count in counts.items())
            if silent:
                self.statusBar().showMessage(f"Сохранено. Состояние БД: {text}")
            else:
                QMessageBox.information(self, "Состояние БД", text)
        except Exception as exc:
            QMessageBox.warning(self, "Состояние БД", str(exc))

    def refresh_parking(self) -> None:
        try:
            rows = self.storage.list_current_parking()
            self.parking_table.setRowCount(0)
            for item in rows:
                row = self.parking_table.rowCount()
                self.parking_table.insertRow(row)
                values = [
                    item.get("session_id"),
                    item.get("entry_at"),
                    item.get("visitor_code"),
                    item.get("plate_text"),
                    item.get("vehicle_make"),
                    item.get("vehicle_model"),
                    item.get("color"),
                    item.get("visitor_full_name") or item.get("visitor_name"),
                    item.get("visit_purpose"),
                ]
                for col, value in enumerate(values):
                    self.parking_table.setItem(row, col, QTableWidgetItem(str(value or "")))
            self.statusBar().showMessage(f"Машин на парковке: {len(rows)}")
        except Exception as exc:
            QMessageBox.warning(self, "Парковка", str(exc))

    def confirm_vehicle_exit(self) -> None:
        row = self.parking_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Выезд", "Выберите машину во вкладке «На парковке».")
            return
        session_item = self.parking_table.item(row, 0)
        plate_item = self.parking_table.item(row, 3)
        if session_item is None:
            QMessageBox.warning(self, "Выезд", "Не найден ID пребывания.")
            return
        plate = plate_item.text() if plate_item is not None else ""
        answer = QMessageBox.question(
            self,
            "Подтверждение выезда",
            f"Подтвердить выезд автомобиля {plate} с территории?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            event = self.storage.record_exit(
                session_id=int(session_item.text()),
                operator_name=self.operator_login,
                note="Выезд подтвержден оператором.",
            )
            self._append_table_row(
                str(event["entry_at"]),
                event["plate_text"],
                "exit",
                event["detector_source"],
                0.0,
            )
            self.refresh_parking()
            self.refresh_audit()
            self.statusBar().showMessage(f"Выезд подтвержден: {event['plate_text']}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка выезда", str(exc))

    def refresh_audit(self) -> None:
        try:
            rows = self.storage.list_parking_audit()
            self.audit_table.setRowCount(0)
            if rows:
                for item in rows:
                    row = self.audit_table.rowCount()
                    self.audit_table.insertRow(row)
                    values = [
                        item.get("entered_at"),
                        item.get("exited_at"),
                        self._format_duration(item.get("duration_seconds")),
                        item.get("visitor_code"),
                        item.get("plate_text"),
                        item.get("vehicle_make"),
                        item.get("vehicle_model"),
                        item.get("visitor_full_name") or item.get("visitor_name"),
                        item.get("exit_operator_full_name")
                        or item.get("exit_operator_login")
                        or item.get("entry_operator_full_name")
                        or item.get("entry_operator_login"),
                    ]
                    for col, value in enumerate(values):
                        self.audit_table.setItem(row, col, QTableWidgetItem(str(value or "")))
                self.statusBar().showMessage(f"Закрытых парковочных сессий: {len(rows)}")
                return

            events = self.storage.list_access_audit()
            for item in events:
                row = self.audit_table.rowCount()
                self.audit_table.insertRow(row)
                values = [
                    item.get("entry_at"),
                    item.get("event_type"),
                    self._decision_to_ru(item.get("decision") or ""),
                    item.get("visitor_code"),
                    item.get("plate_text"),
                    item.get("vehicle_make"),
                    item.get("vehicle_model"),
                    item.get("visitor_full_name") or item.get("visitor_name"),
                    item.get("operator_full_name") or item.get("operator_login"),
                ]
                for col, value in enumerate(values):
                    self.audit_table.setItem(row, col, QTableWidgetItem(str(value or "")))
            self.statusBar().showMessage(f"Событий аудита: {len(events)}")
        except Exception as exc:
            QMessageBox.warning(self, "Аудит", str(exc))

    def _format_duration(self, seconds: object) -> str:
        if seconds is None:
            return ""
        total = int(seconds)
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _decision_to_ru(self, decision: str) -> str:
        if decision == "exit":
            return "выезд"
        return "разрешено" if decision == "allowed" else "отказ"

    def _operator_display_name(self) -> str:
        if self.operator_full_name:
            return f"{self.operator_login} - {self.operator_full_name}"
        return self.operator_login

    def _append_table_row(self, ts: str, plate: str, decision: str, detector: str, ocr_conf: float) -> None:
        return

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.storage.close()
        super().closeEvent(event)


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
