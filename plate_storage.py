from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime, timezone
from typing import Any

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError as exc:
    raise SystemExit(
        "psycopg is not installed. Install it with: python -m pip install psycopg[binary]"
    ) from exc


DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/plates_audit"
PLATE_RE = re.compile(r"[^0-9A-Z]")
VISITOR_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
DEFAULT_VEHICLE_CATALOG = {
    "Toyota": ["Camry", "Corolla", "RAV4", "Land Cruiser", "Prado"],
    "Volkswagen": ["Polo", "Passat", "Tiguan", "Touareg", "Jetta"],
    "Kia": ["Rio", "Sportage", "Ceed", "Sorento", "K5"],
    "Hyundai": ["Solaris", "Creta", "Tucson", "Santa Fe", "Elantra"],
    "Lada": ["Granta", "Vesta", "Niva", "Largus", "Priora"],
    "Mercedes-Benz": ["E-Class", "C-Class", "S-Class", "GLC", "GLE"],
    "BMW": ["3 Series", "5 Series", "X3", "X5", "X6"],
    "Audi": ["A4", "A6", "Q5", "Q7", "A3"],
    "Renault": ["Logan", "Duster", "Sandero", "Kaptur", "Arkana"],
    "Nissan": ["Qashqai", "X-Trail", "Almera", "Murano", "Teana"],
    "Ford": ["Focus", "Mondeo", "Kuga", "Transit", "Explorer"],
    "Skoda": ["Octavia", "Rapid", "Kodiaq", "Karoq", "Superb"],
    "Chevrolet": ["Niva", "Cruze", "Aveo", "Captiva", "Tahoe"],
    "Mazda": ["3", "6", "CX-5", "CX-7", "CX-9"],
    "Mitsubishi": ["Outlander", "Pajero", "Lancer", "ASX", "Eclipse Cross"],
    "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Fit"],
    "Subaru": ["Forester", "Outback", "XV", "Legacy", "Impreza"],
    "Volvo": ["XC60", "XC90", "S60", "S90", "V60"],
    "Geely": ["Coolray", "Atlas", "Tugella", "Monjaro", "Emgrand"],
    "Haval": ["Jolion", "F7", "Dargo", "H9", "M6"],
}


def normalize_plate(value: str | None) -> str:
    if not value:
        return ""
    text = value.strip().upper().replace(" ", "").replace("-", "")
    substitutions = str.maketrans(
        {
            "А": "A",
            "В": "B",
            "С": "C",
            "Е": "E",
            "Н": "H",
            "К": "K",
            "М": "M",
            "О": "O",
            "Р": "P",
            "Т": "T",
            "Х": "X",
            "У": "Y",
        }
    )
    return PLATE_RE.sub("", text.translate(substitutions))


class PlateStorage:
    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.getenv("PLATE_DB_DSN") or DEFAULT_DSN
        self.conn = psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn.cursor() as cur:
            self._create_core_tables(cur)
            self._migrate_existing_tables(cur)
            self._create_compat_indexes(cur)
            self._seed_default_gate(cur)
            self._merge_duplicate_persons(cur)

    def _create_core_tables(self, cur) -> None:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id BIGSERIAL PRIMARY KEY,
                full_name TEXT NOT NULL,
                visitor_code TEXT UNIQUE,
                phone TEXT,
                company TEXT,
                document_no TEXT,
                comment TEXT,
                is_active BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS operators (
                id BIGSERIAL PRIMARY KEY,
                login TEXT NOT NULL UNIQUE,
                full_name TEXT,
                is_active BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS gates (
                id BIGSERIAL PRIMARY KEY,
                code TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                direction TEXT NOT NULL DEFAULT 'entry'
                    CHECK (direction IN ('entry', 'exit', 'both')),
                is_active BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicle_makes (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                is_active BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicle_models (
                id BIGSERIAL PRIMARY KEY,
                make_id BIGINT NOT NULL REFERENCES vehicle_makes(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(make_id, name)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicles (
                id BIGSERIAL PRIMARY KEY,
                primary_plate_text TEXT NOT NULL UNIQUE,
                make_id BIGINT REFERENCES vehicle_makes(id) ON DELETE SET NULL,
                model_id BIGINT REFERENCES vehicle_models(id) ON DELETE SET NULL,
                vehicle_make TEXT,
                vehicle_model TEXT,
                color TEXT,
                manufacture_year SMALLINT
                    CHECK (manufacture_year IS NULL OR manufacture_year BETWEEN 1950 AND 2100),
                vin TEXT,
                body_number TEXT,
                owner_person_id BIGINT REFERENCES persons(id) ON DELETE SET NULL,
                access_status TEXT NOT NULL DEFAULT 'unknown'
                    CHECK (access_status IN ('allowed', 'blocked', 'unknown')),
                comment TEXT,
                is_active BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicle_person_links (
                id BIGSERIAL PRIMARY KEY,
                vehicle_id BIGINT NOT NULL REFERENCES vehicles(id) ON DELETE CASCADE,
                person_id BIGINT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                relation_type TEXT NOT NULL
                    CHECK (relation_type IN ('owner', 'visitor', 'driver')),
                valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
                valid_to TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(vehicle_id, person_id, relation_type, valid_from),
                CHECK (valid_to IS NULL OR valid_to > valid_from)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicle_plate_aliases (
                id BIGSERIAL PRIMARY KEY,
                vehicle_id BIGINT NOT NULL REFERENCES vehicles(id) ON DELETE CASCADE,
                plate_text TEXT NOT NULL UNIQUE,
                is_primary BOOLEAN NOT NULL DEFAULT false,
                valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
                valid_to TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CHECK (valid_to IS NULL OR valid_to > valid_from)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicle_access_rules (
                id BIGSERIAL PRIMARY KEY,
                vehicle_id BIGINT NOT NULL REFERENCES vehicles(id) ON DELETE CASCADE,
                gate_id BIGINT REFERENCES gates(id) ON DELETE CASCADE,
                rule_type TEXT NOT NULL CHECK (rule_type IN ('allow', 'deny')),
                starts_at TIMESTAMPTZ,
                ends_at TIMESTAMPTZ,
                reason TEXT,
                is_active BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CHECK (ends_at IS NULL OR starts_at IS NULL OR ends_at > starts_at)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS access_events (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                entry_at TIMESTAMPTZ NOT NULL,
                gate_id BIGINT REFERENCES gates(id) ON DELETE SET NULL,
                vehicle_id BIGINT REFERENCES vehicles(id) ON DELETE SET NULL,
                operator_id BIGINT REFERENCES operators(id) ON DELETE SET NULL,
                detected_plate_text TEXT NOT NULL,
                plate_text TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                event_type TEXT NOT NULL DEFAULT 'entry'
                    CHECK (event_type IN ('entry', 'exit')),
                decision TEXT NOT NULL CHECK (decision IN ('allowed', 'denied')),
                decision_source TEXT NOT NULL DEFAULT 'operator'
                    CHECK (decision_source IN ('operator', 'auto_rule', 'auto_unknown')),
                visitor_name TEXT,
                visitor_code TEXT,
                visitor_person_id BIGINT REFERENCES persons(id) ON DELETE SET NULL,
                visit_purpose TEXT,
                note TEXT,
                detector_source TEXT NOT NULL,
                detector_score DOUBLE PRECISION NOT NULL,
                ocr_confidence DOUBLE PRECISION NOT NULL,
                bbox_json JSONB NOT NULL,
                image_path TEXT,
                vis_path TEXT,
                crop_path TEXT,
                plate_image_path TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS parking_sessions (
                id BIGSERIAL PRIMARY KEY,
                vehicle_id BIGINT NOT NULL REFERENCES vehicles(id) ON DELETE CASCADE,
                entry_event_id BIGINT REFERENCES access_events(id) ON DELETE SET NULL,
                exit_event_id BIGINT REFERENCES access_events(id) ON DELETE SET NULL,
                visitor_person_id BIGINT REFERENCES persons(id) ON DELETE SET NULL,
                visitor_code TEXT,
                entered_at TIMESTAMPTZ NOT NULL,
                exited_at TIMESTAMPTZ,
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'closed')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CHECK (exited_at IS NULL OR exited_at >= entered_at)
            )
            """
        )

    def _migrate_existing_tables(self, cur) -> None:
        self._add_column(cur, "persons", "visitor_code", "TEXT")
        self._drop_not_null_if_column_exists(cur, "vehicles", "plate_text")
        self._add_column(cur, "vehicles", "primary_plate_text", "TEXT")
        self._add_column(cur, "vehicles", "make_id", "BIGINT")
        self._add_column(cur, "vehicles", "model_id", "BIGINT")
        self._add_column(cur, "vehicles", "vehicle_make", "TEXT")
        self._add_column(cur, "vehicles", "vehicle_model", "TEXT")
        self._add_column(cur, "vehicles", "color", "TEXT")
        self._add_column(cur, "vehicles", "manufacture_year", "SMALLINT")
        self._add_column(cur, "vehicles", "vin", "TEXT")
        self._add_column(cur, "vehicles", "body_number", "TEXT")
        self._add_column(cur, "vehicles", "owner_person_id", "BIGINT")
        self._add_column(cur, "vehicles", "access_status", "TEXT NOT NULL DEFAULT 'unknown'")
        self._add_column(cur, "vehicles", "comment", "TEXT")
        self._add_column(cur, "vehicles", "is_active", "BOOLEAN NOT NULL DEFAULT true")
        self._add_column(cur, "vehicles", "created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()")
        self._add_column(cur, "vehicles", "updated_at", "TIMESTAMPTZ NOT NULL DEFAULT now()")
        if self._column_exists(cur, "vehicles", "plate_text"):
            cur.execute(
                """
                UPDATE vehicles
                SET primary_plate_text = plate_text
                WHERE primary_plate_text IS NULL
                """
            )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_vehicles_primary_plate_text_all
            ON vehicles(primary_plate_text)
            """
        )
        self._add_column(cur, "access_events", "gate_id", "BIGINT")
        self._add_column(cur, "access_events", "operator_id", "BIGINT")
        self._add_column(cur, "access_events", "event_type", "TEXT NOT NULL DEFAULT 'entry'")
        self._add_column(cur, "access_events", "decision_source", "TEXT NOT NULL DEFAULT 'operator'")
        self._add_column(cur, "access_events", "visitor_name", "TEXT")
        self._add_column(cur, "access_events", "visitor_code", "TEXT")
        self._add_column(cur, "access_events", "visitor_person_id", "BIGINT")
        self._add_column(cur, "access_events", "visit_purpose", "TEXT")
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_persons_visitor_code ON persons(visitor_code) WHERE visitor_code IS NOT NULL"
        )

    def _add_column(self, cur, table_name: str, column_name: str, column_sql: str) -> None:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {column_name} {column_sql}")

    def _drop_not_null_if_column_exists(self, cur, table_name: str, column_name: str) -> None:
        if self._column_exists(cur, table_name, column_name):
            cur.execute(f"ALTER TABLE {table_name} ALTER COLUMN {column_name} DROP NOT NULL")

    def _column_exists(self, cur, table_name: str, column_name: str) -> bool:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
              AND column_name = %s
            """,
            (table_name, column_name),
        )
        return cur.fetchone() is not None

    def _merge_duplicate_persons(self, cur) -> None:
        cur.execute(
            """
            SELECT lower(trim(full_name)) AS key
            FROM persons
            WHERE trim(full_name) <> ''
            GROUP BY lower(trim(full_name))
            HAVING COUNT(*) > 1
            """
        )
        duplicate_keys = [row["key"] for row in cur.fetchall()]
        for key in duplicate_keys:
            cur.execute(
                """
                SELECT *
                FROM persons
                WHERE lower(trim(full_name)) = %s
                ORDER BY
                    (phone IS NOT NULL) DESC,
                    (visitor_code IS NOT NULL) DESC,
                    id ASC
                """,
                (key,),
            )
            rows = cur.fetchall()
            if len(rows) < 2:
                continue

            target = rows[0]
            source_rows = rows[1:]
            merged_phone = target["phone"] or next((row["phone"] for row in source_rows if row["phone"]), None)
            merged_code = target["visitor_code"] or next((row["visitor_code"] for row in source_rows if row["visitor_code"]), None)
            cur.execute(
                """
                UPDATE persons
                SET phone = %s, visitor_code = %s, updated_at = now()
                WHERE id = %s
                """,
                (merged_phone, merged_code, target["id"]),
            )

            for source in source_rows:
                source_id = source["id"]
                cur.execute(
                    "UPDATE vehicles SET owner_person_id = %s WHERE owner_person_id = %s",
                    (target["id"], source_id),
                )
                cur.execute(
                    "UPDATE access_events SET visitor_person_id = %s WHERE visitor_person_id = %s",
                    (target["id"], source_id),
                )
                cur.execute(
                    "UPDATE parking_sessions SET visitor_person_id = %s WHERE visitor_person_id = %s",
                    (target["id"], source_id),
                )
                cur.execute(
                    """
                    DELETE FROM vehicle_person_links src
                    WHERE src.person_id = %s
                      AND EXISTS (
                          SELECT 1
                          FROM vehicle_person_links dst
                          WHERE dst.vehicle_id = src.vehicle_id
                            AND dst.person_id = %s
                            AND dst.relation_type = src.relation_type
                            AND dst.valid_to IS NULL
                      )
                    """,
                    (source_id, target["id"]),
                )
                cur.execute(
                    "UPDATE vehicle_person_links SET person_id = %s WHERE person_id = %s",
                    (target["id"], source_id),
                )
                cur.execute("DELETE FROM persons WHERE id = %s", (source_id,))

    def _create_compat_indexes(self, cur) -> None:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_persons_phone ON persons(phone) WHERE phone IS NOT NULL"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_vehicles_make_model ON vehicles(vehicle_make, vehicle_model)"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_models_make ON vehicle_models(make_id)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_vehicles_owner ON vehicles(owner_person_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_vehicle_person_links_vehicle ON vehicle_person_links(vehicle_id, relation_type)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_vehicle_person_links_person ON vehicle_person_links(person_id, relation_type)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_vehicle_plate_aliases_vehicle ON vehicle_plate_aliases(vehicle_id)"
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_vehicle_access_rules_active
            ON vehicle_access_rules(vehicle_id, gate_id, rule_type)
            WHERE is_active = true
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_access_events_entry_at ON access_events(entry_at DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_access_events_plate_time ON access_events(plate_text, entry_at DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_access_events_vehicle_time ON access_events(vehicle_id, entry_at DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_access_events_gate_time ON access_events(gate_id, entry_at DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_access_events_decision_time ON access_events(decision, entry_at DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_access_events_type_time ON access_events(event_type, entry_at DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_parking_sessions_active ON parking_sessions(status, entered_at DESC)"
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_parking_sessions_active_vehicle ON parking_sessions(vehicle_id) WHERE status = 'active'"
        )

    def _seed_default_gate(self, cur) -> None:
        cur.execute(
            """
            INSERT INTO gates(code, name, direction)
            VALUES ('MAIN_ENTRY', 'Главный въезд', 'entry')
            ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name
            """
        )
        cur.execute(
            """
            INSERT INTO gates(code, name, direction)
            VALUES ('MAIN_EXIT', 'Главный выезд', 'exit')
            ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name
            """
        )
        for make, models in DEFAULT_VEHICLE_CATALOG.items():
            cur.execute(
                """
                INSERT INTO vehicle_makes(name)
                VALUES (%s)
                ON CONFLICT (name) DO UPDATE SET is_active = true
                RETURNING id
                """,
                (make,),
            )
            make_id = cur.fetchone()["id"]
            for model in models:
                cur.execute(
                    """
                    INSERT INTO vehicle_models(make_id, name)
                    VALUES (%s, %s)
                    ON CONFLICT (make_id, name) DO UPDATE SET is_active = true
                    """,
                    (make_id, model),
                )

    def authenticate_operator(self, login: str, full_name: str = "") -> dict[str, Any]:
        login_value = login.strip()
        if not login_value:
            raise ValueError("Operator login is empty.")
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO operators(login, full_name)
                    VALUES (%s, NULLIF(%s, ''))
                    ON CONFLICT (login) DO UPDATE SET
                        full_name = COALESCE(NULLIF(EXCLUDED.full_name, ''), operators.full_name),
                        is_active = true
                    RETURNING *
                    """,
                    (login_value, full_name.strip()),
                )
                return cur.fetchone()

    def get_operator(self, login: str) -> dict[str, Any] | None:
        login_value = login.strip()
        if not login_value:
            return None
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM operators
                WHERE login = %s AND is_active = true
                """,
                (login_value,),
            )
            return cur.fetchone()

    def list_vehicle_makes(self) -> list[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT name FROM vehicle_makes WHERE is_active = true ORDER BY name"
            )
            return [row["name"] for row in cur.fetchall()]

    def list_vehicle_models(self, make_name: str = "") -> list[str]:
        with self.conn.cursor() as cur:
            if make_name.strip():
                cur.execute(
                    """
                    SELECT vm.name
                    FROM vehicle_models vm
                    JOIN vehicle_makes mk ON mk.id = vm.make_id
                    WHERE vm.is_active = true AND mk.name = %s
                    ORDER BY vm.name
                    """,
                    (make_name.strip(),),
                )
            else:
                cur.execute(
                    "SELECT name FROM vehicle_models WHERE is_active = true ORDER BY name"
                )
            return [row["name"] for row in cur.fetchall()]

    def generate_visitor_code(self) -> str:
        with self.conn.cursor() as cur:
            while True:
                code = "".join(secrets.choice(VISITOR_CODE_ALPHABET) for _ in range(6))
                cur.execute("SELECT 1 FROM persons WHERE visitor_code = %s", (code,))
                if cur.fetchone() is None:
                    return code

    def find_vehicle(self, plate_text: str) -> dict[str, Any] | None:
        plate = normalize_plate(plate_text)
        if not plate:
            return None
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    v.*,
                    p.full_name AS owner_name,
                    p.phone AS owner_phone,
                    p.company AS owner_company,
                    p.document_no AS owner_document_no
                FROM vehicles v
                LEFT JOIN persons p ON p.id = v.owner_person_id
                LEFT JOIN vehicle_plate_aliases a ON a.vehicle_id = v.id
                WHERE v.primary_plate_text = %s OR a.plate_text = %s
                ORDER BY (v.primary_plate_text = %s) DESC, a.is_primary DESC NULLS LAST
                LIMIT 1
                """,
                (plate, plate, plate),
            )
            return cur.fetchone()

    def get_last_vehicle_visitor(self, vehicle_id: int) -> dict[str, Any] | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COALESCE(visitor.full_name, e.visitor_name) AS visitor_name,
                    visitor.phone AS visitor_phone,
                    COALESCE(ps.visitor_code, e.visitor_code, visitor.visitor_code) AS visitor_code,
                    e.visit_purpose
                FROM access_events e
                LEFT JOIN parking_sessions ps ON ps.entry_event_id = e.id
                LEFT JOIN persons visitor ON visitor.id = COALESCE(ps.visitor_person_id, e.visitor_person_id)
                WHERE e.vehicle_id = %s
                  AND e.event_type = 'entry'
                  AND e.decision = 'allowed'
                ORDER BY
                    (ps.status = 'active') DESC NULLS LAST,
                    e.entry_at DESC,
                    e.id DESC
                LIMIT 1
                """,
                (vehicle_id,),
            )
            return cur.fetchone()

    def get_last_entry(self, plate_text: str) -> dict[str, Any] | None:
        plate = normalize_plate(plate_text)
        if not plate:
            return None
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.*,
                    g.name AS gate_name,
                    o.login AS operator_login,
                    o.full_name AS operator_full_name
                FROM access_events e
                LEFT JOIN gates g ON g.id = e.gate_id
                LEFT JOIN operators o ON o.id = e.operator_id
                WHERE e.plate_text = %s
                ORDER BY e.entry_at DESC, e.id DESC
                LIMIT 1
                """,
                (plate,),
            )
            return cur.fetchone()

    def build_review_context(self, plate_text: str) -> dict[str, Any]:
        plate = normalize_plate(plate_text)
        vehicle = self.find_vehicle(plate)
        return {
            "plate_text": plate,
            "vehicle": vehicle,
            "visitor": self.get_last_vehicle_visitor(vehicle["id"]) if vehicle else None,
            "last_entry": self.get_last_entry(plate),
            "active_rule": self.get_active_rule(vehicle["id"]) if vehicle else None,
        }

    def get_active_rule(self, vehicle_id: int, gate_code: str = "MAIN_ENTRY") -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.*
                FROM vehicle_access_rules r
                LEFT JOIN gates g ON g.id = r.gate_id
                WHERE r.vehicle_id = %s
                  AND r.is_active = true
                  AND (r.gate_id IS NULL OR g.code = %s)
                  AND (r.starts_at IS NULL OR r.starts_at <= %s)
                  AND (r.ends_at IS NULL OR r.ends_at >= %s)
                ORDER BY
                    (r.rule_type = 'deny') DESC,
                    (r.gate_id IS NOT NULL) DESC,
                    r.id DESC
                LIMIT 1
                """,
                (vehicle_id, gate_code, now, now),
            )
            return cur.fetchone()

    def record_operator_decision(
        self,
        *,
        detected_plate_text: str,
        plate_text: str,
        raw_text: str,
        decision: str,
        operator_name: str = "",
        visitor_name: str = "",
        visitor_phone: str = "",
        visitor_code: str = "",
        visit_purpose: str = "",
        owner_name: str = "",
        owner_phone: str = "",
        vehicle_make: str = "",
        vehicle_model: str = "",
        color: str = "",
        manufacture_year: int | None = None,
        comment: str = "",
        note: str = "",
        detector_source: str,
        detector_score: float,
        ocr_confidence: float,
        bbox: tuple[int, int, int, int],
        image_path: str | None = None,
        vis_path: str | None = None,
        crop_path: str | None = None,
        plate_image_path: str | None = None,
        entry_at: datetime | None = None,
        gate_code: str = "MAIN_ENTRY",
    ) -> dict[str, Any]:
        plate = normalize_plate(plate_text)
        detected_plate = normalize_plate(detected_plate_text)
        if not plate:
            raise ValueError("Plate number is empty.")
        if decision not in {"allowed", "denied"}:
            raise ValueError("Decision must be 'allowed' or 'denied'.")

        entry_time = entry_at or datetime.now(timezone.utc)
        bbox_json = json.dumps({"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]})

        with self.conn.transaction():
            with self.conn.cursor() as cur:
                gate_id = self._get_gate_id(cur, gate_code)
                operator_id = self._get_or_create_operator(cur, operator_name)
                owner_id = self._get_or_create_person(cur, owner_name, owner_phone)
                visitor_id = self._get_or_create_person(
                    cur,
                    visitor_name,
                    visitor_phone,
                    visitor_code=visitor_code,
                )
                make_id, model_id = self._get_or_create_make_model(
                    cur, vehicle_make, vehicle_model
                )
                vehicle_id = self._get_or_create_vehicle(
                    cur,
                    plate_text=plate,
                    owner_person_id=owner_id,
                    make_id=make_id,
                    model_id=model_id,
                    vehicle_make=vehicle_make,
                    vehicle_model=vehicle_model,
                    color=color,
                    manufacture_year=manufacture_year,
                    comment=comment,
                )
                self._ensure_plate_alias(cur, vehicle_id, plate, is_primary=True)
                if owner_id is not None:
                    self._ensure_vehicle_person_link(cur, vehicle_id, owner_id, "owner")
                if visitor_id is not None:
                    self._ensure_vehicle_person_link(cur, vehicle_id, visitor_id, "visitor")

                cur.execute(
                    """
                    INSERT INTO access_events (
                        entry_at, gate_id, vehicle_id, operator_id,
                        detected_plate_text, plate_text, raw_text, event_type, decision,
                        visitor_name, visitor_code, visitor_person_id, visit_purpose, note,
                        detector_source, detector_score, ocr_confidence, bbox_json,
                        image_path, vis_path, crop_path, plate_image_path
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, 'entry', %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s::jsonb,
                        %s, %s, %s, %s
                    )
                    RETURNING *
                    """,
                    (
                        entry_time,
                        gate_id,
                        vehicle_id,
                        operator_id,
                        detected_plate,
                        plate,
                        raw_text,
                        decision,
                        visitor_name,
                        visitor_code.strip() or None,
                        visitor_id,
                        visit_purpose,
                        note,
                        detector_source,
                        float(detector_score),
                        float(ocr_confidence),
                        bbox_json,
                        image_path,
                        vis_path,
                        crop_path,
                        plate_image_path,
                    ),
                )
                event = cur.fetchone()
                if decision == "allowed":
                    self._open_parking_session(
                        cur,
                        vehicle_id=vehicle_id,
                        entry_event_id=event["id"],
                        visitor_person_id=visitor_id,
                        visitor_code=visitor_code,
                        entered_at=entry_time,
                    )
                return event

    def _get_gate_id(self, cur, gate_code: str) -> int:
        cur.execute("SELECT id FROM gates WHERE code = %s", (gate_code,))
        row = cur.fetchone()
        if row:
            return row["id"]
        cur.execute(
            "INSERT INTO gates(code, name, direction) VALUES (%s, %s, 'entry') RETURNING id",
            (gate_code, gate_code),
        )
        return cur.fetchone()["id"]

    def _get_or_create_make_model(
        self, cur, vehicle_make: str, vehicle_model: str
    ) -> tuple[int | None, int | None]:
        make = vehicle_make.strip()
        model = vehicle_model.strip()
        if not make:
            return None, None
        cur.execute(
            """
            INSERT INTO vehicle_makes(name)
            VALUES (%s)
            ON CONFLICT (name) DO UPDATE SET is_active = true
            RETURNING id
            """,
            (make,),
        )
        make_id = cur.fetchone()["id"]
        if not model:
            return make_id, None
        cur.execute(
            """
            INSERT INTO vehicle_models(make_id, name)
            VALUES (%s, %s)
            ON CONFLICT (make_id, name) DO UPDATE SET is_active = true
            RETURNING id
            """,
            (make_id, model),
        )
        return make_id, cur.fetchone()["id"]

    def _get_or_create_operator(self, cur, operator_name: str) -> int | None:
        login = operator_name.strip()
        if not login:
            return None
        cur.execute(
            """
            INSERT INTO operators(login, full_name)
            VALUES (%s, %s)
            ON CONFLICT (login) DO UPDATE SET full_name = COALESCE(operators.full_name, EXCLUDED.full_name)
            RETURNING id
            """,
            (login, login),
        )
        return cur.fetchone()["id"]

    def _get_or_create_person(
        self,
        cur,
        full_name: str,
        phone: str,
        visitor_code: str = "",
    ) -> int | None:
        name = full_name.strip()
        phone_value = phone.strip()
        code_value = visitor_code.strip().upper()
        if not name and not phone_value and not code_value:
            return None
        if code_value:
            cur.execute("SELECT id FROM persons WHERE visitor_code = %s LIMIT 1", (code_value,))
            row = cur.fetchone()
            if row:
                cur.execute(
                    """
                    UPDATE persons
                    SET
                        full_name = COALESCE(NULLIF(%s, ''), full_name),
                        phone = COALESCE(NULLIF(%s, ''), phone),
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (name, phone_value, row["id"]),
                )
                return row["id"]
        if phone_value:
            cur.execute("SELECT id FROM persons WHERE phone = %s LIMIT 1", (phone_value,))
            row = cur.fetchone()
            if row:
                if name:
                    cur.execute(
                        """
                        UPDATE persons
                        SET
                            full_name = %s,
                            visitor_code = COALESCE(NULLIF(%s, ''), visitor_code),
                            updated_at = now()
                        WHERE id = %s
                        """,
                        (name, code_value, row["id"]),
                    )
                return row["id"]
        if name:
            cur.execute(
                """
                SELECT id
                FROM persons
                WHERE lower(trim(full_name)) = lower(trim(%s))
                ORDER BY
                    (phone IS NOT NULL) DESC,
                    (visitor_code IS NOT NULL) DESC,
                    id ASC
                LIMIT 1
                """,
                (name,),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    """
                    UPDATE persons
                    SET
                        phone = COALESCE(NULLIF(%s, ''), phone),
                        visitor_code = COALESCE(NULLIF(%s, ''), visitor_code),
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (phone_value, code_value, row["id"]),
                )
                return row["id"]
        cur.execute(
            """
            INSERT INTO persons(full_name, phone, visitor_code)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (name or (f"Посетитель {code_value}" if code_value else "Не указано"), phone_value or None, code_value or None),
        )
        return cur.fetchone()["id"]

    def _open_parking_session(
        self,
        cur,
        *,
        vehicle_id: int,
        entry_event_id: int,
        visitor_person_id: int | None,
        visitor_code: str,
        entered_at: datetime,
    ) -> None:
        cur.execute(
            """
            UPDATE parking_sessions
            SET status = 'closed', exited_at = COALESCE(exited_at, %s), updated_at = now()
            WHERE vehicle_id = %s AND status = 'active'
            """,
            (entered_at, vehicle_id),
        )
        cur.execute(
            """
            INSERT INTO parking_sessions (
                vehicle_id, entry_event_id, visitor_person_id, visitor_code, entered_at
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (vehicle_id, entry_event_id, visitor_person_id, visitor_code.strip() or None, entered_at),
        )

    def _get_or_create_vehicle(
        self,
        cur,
        *,
        plate_text: str,
        owner_person_id: int | None,
        make_id: int | None,
        model_id: int | None,
        vehicle_make: str,
        vehicle_model: str,
        color: str,
        manufacture_year: int | None,
        comment: str,
    ) -> int:
        cur.execute(
            """
            INSERT INTO vehicles (
                primary_plate_text, owner_person_id, make_id, model_id, vehicle_make, vehicle_model,
                color, manufacture_year, comment
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (primary_plate_text) DO UPDATE SET
                owner_person_id = COALESCE(EXCLUDED.owner_person_id, vehicles.owner_person_id),
                make_id = COALESCE(EXCLUDED.make_id, vehicles.make_id),
                model_id = COALESCE(EXCLUDED.model_id, vehicles.model_id),
                vehicle_make = COALESCE(NULLIF(EXCLUDED.vehicle_make, ''), vehicles.vehicle_make),
                vehicle_model = COALESCE(NULLIF(EXCLUDED.vehicle_model, ''), vehicles.vehicle_model),
                color = COALESCE(NULLIF(EXCLUDED.color, ''), vehicles.color),
                manufacture_year = COALESCE(EXCLUDED.manufacture_year, vehicles.manufacture_year),
                comment = COALESCE(NULLIF(EXCLUDED.comment, ''), vehicles.comment),
                updated_at = now()
            RETURNING id
            """,
            (
                plate_text,
                owner_person_id,
                make_id,
                model_id,
                vehicle_make,
                vehicle_model,
                color,
                manufacture_year,
                comment,
            ),
        )
        return cur.fetchone()["id"]

    def _ensure_plate_alias(self, cur, vehicle_id: int, plate_text: str, is_primary: bool) -> None:
        cur.execute(
            """
            INSERT INTO vehicle_plate_aliases(vehicle_id, plate_text, is_primary)
            VALUES (%s, %s, %s)
            ON CONFLICT (plate_text) DO UPDATE SET
                vehicle_id = EXCLUDED.vehicle_id,
                is_primary = EXCLUDED.is_primary,
                valid_to = NULL
            """,
            (vehicle_id, plate_text, is_primary),
        )

    def _ensure_vehicle_person_link(
        self,
        cur,
        vehicle_id: int,
        person_id: int,
        relation_type: str,
    ) -> None:
        cur.execute(
            """
            SELECT id
            FROM vehicle_person_links
            WHERE vehicle_id = %s
              AND person_id = %s
              AND relation_type = %s
              AND valid_to IS NULL
            LIMIT 1
            """,
            (vehicle_id, person_id, relation_type),
        )
        if cur.fetchone():
            return
        cur.execute(
            """
            INSERT INTO vehicle_person_links(vehicle_id, person_id, relation_type)
            VALUES (%s, %s, %s)
            """,
            (vehicle_id, person_id, relation_type),
        )

    def insert_read(
        self,
        image_path: str,
        plate_text: str,
        raw_text: str,
        detector_source: str,
        detector_score: float,
        ocr_confidence: float,
        bbox: tuple[int, int, int, int],
        vis_path: str | None = None,
        crop_path: str | None = None,
        plate_image_path: str | None = None,
    ) -> None:
        self.record_operator_decision(
            detected_plate_text=plate_text,
            plate_text=plate_text,
            raw_text=raw_text,
            decision="allowed",
            detector_source=detector_source,
            detector_score=detector_score,
            ocr_confidence=ocr_confidence,
            bbox=bbox,
            image_path=image_path,
            vis_path=vis_path,
            crop_path=crop_path,
            plate_image_path=plate_image_path,
            note="Recorded automatically without operator dialog.",
        )

    def recent_reads(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.*,
                    v.vehicle_make,
                    v.vehicle_model,
                    v.color,
                    p.full_name AS owner_name,
                    g.name AS gate_name
                FROM access_events e
                LEFT JOIN vehicles v ON v.id = e.vehicle_id
                LEFT JOIN persons p ON p.id = v.owner_person_id
                LEFT JOIN gates g ON g.id = e.gate_id
                ORDER BY e.entry_at DESC, e.id DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            return list(cur.fetchall())

    def list_current_parking(self) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ps.id AS session_id,
                    ps.vehicle_id,
                    ps.entered_at AS entry_at,
                    ps.visitor_code,
                    e.plate_text,
                    e.visitor_name,
                    e.visit_purpose,
                    v.vehicle_make,
                    v.vehicle_model,
                    v.color,
                    p.full_name AS owner_name,
                    vp.full_name AS visitor_full_name,
                    owner_link.id AS owner_link_id,
                    visitor_link.id AS visitor_link_id,
                    g.name AS gate_name
                FROM parking_sessions ps
                JOIN vehicles v ON v.id = ps.vehicle_id
                LEFT JOIN access_events e ON e.id = ps.entry_event_id
                LEFT JOIN persons p ON p.id = v.owner_person_id
                LEFT JOIN persons vp ON vp.id = ps.visitor_person_id
                LEFT JOIN vehicle_person_links owner_link
                    ON owner_link.vehicle_id = v.id
                   AND owner_link.person_id = p.id
                   AND owner_link.relation_type = 'owner'
                   AND owner_link.valid_to IS NULL
                LEFT JOIN vehicle_person_links visitor_link
                    ON visitor_link.vehicle_id = v.id
                   AND visitor_link.person_id = vp.id
                   AND visitor_link.relation_type = 'visitor'
                   AND visitor_link.valid_to IS NULL
                LEFT JOIN gates g ON g.id = e.gate_id
                WHERE ps.status = 'active'
                ORDER BY ps.entered_at DESC
                """
            )
            return list(cur.fetchall())

    def list_parking_audit(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ps.id AS session_id,
                    ps.entered_at,
                    ps.exited_at,
                    ps.visitor_code,
                    EXTRACT(EPOCH FROM (ps.exited_at - ps.entered_at))::BIGINT AS duration_seconds,
                    entry_event.plate_text,
                    entry_event.visitor_name,
                    entry_event.visit_purpose,
                    v.vehicle_make,
                    v.vehicle_model,
                    v.color,
                    owner.full_name AS owner_name,
                    visitor.full_name AS visitor_full_name,
                    entry_gate.name AS entry_gate_name,
                    exit_gate.name AS exit_gate_name,
                    entry_operator.login AS entry_operator_login,
                    entry_operator.full_name AS entry_operator_full_name,
                    exit_operator.login AS exit_operator_login,
                    exit_operator.full_name AS exit_operator_full_name
                FROM parking_sessions ps
                JOIN vehicles v ON v.id = ps.vehicle_id
                LEFT JOIN access_events entry_event ON entry_event.id = ps.entry_event_id
                LEFT JOIN access_events exit_event ON exit_event.id = ps.exit_event_id
                LEFT JOIN persons owner ON owner.id = v.owner_person_id
                LEFT JOIN persons visitor ON visitor.id = ps.visitor_person_id
                LEFT JOIN gates entry_gate ON entry_gate.id = entry_event.gate_id
                LEFT JOIN gates exit_gate ON exit_gate.id = exit_event.gate_id
                LEFT JOIN operators entry_operator ON entry_operator.id = entry_event.operator_id
                LEFT JOIN operators exit_operator ON exit_operator.id = exit_event.operator_id
                WHERE ps.status = 'closed'
                ORDER BY ps.exited_at DESC NULLS LAST, ps.entered_at DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            return list(cur.fetchall())

    def list_access_audit(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.entry_at,
                    e.event_type,
                    e.decision,
                    e.plate_text,
                    e.visitor_code,
                    e.visitor_name,
                    e.visit_purpose,
                    v.vehicle_make,
                    v.vehicle_model,
                    owner.full_name AS owner_name,
                    visitor.full_name AS visitor_full_name,
                    g.name AS gate_name,
                    o.login AS operator_login,
                    o.full_name AS operator_full_name
                FROM access_events e
                LEFT JOIN vehicles v ON v.id = e.vehicle_id
                LEFT JOIN persons owner ON owner.id = v.owner_person_id
                LEFT JOIN persons visitor ON visitor.id = e.visitor_person_id
                LEFT JOIN gates g ON g.id = e.gate_id
                LEFT JOIN operators o ON o.id = e.operator_id
                ORDER BY e.entry_at DESC, e.id DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            return list(cur.fetchall())

    def record_exit(
        self,
        *,
        session_id: int,
        operator_name: str,
        note: str = "",
        gate_code: str = "MAIN_EXIT",
    ) -> dict[str, Any]:
        exit_time = datetime.now(timezone.utc)
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        ps.*,
                        v.primary_plate_text
                    FROM parking_sessions ps
                    JOIN vehicles v ON v.id = ps.vehicle_id
                    WHERE ps.id = %s AND ps.status = 'active'
                    FOR UPDATE
                    """,
                    (int(session_id),),
                )
                session = cur.fetchone()
                if session is None:
                    raise ValueError("Active parking session was not found.")

                gate_id = self._get_gate_id(cur, gate_code)
                operator_id = self._get_or_create_operator(cur, operator_name)
                plate = session["primary_plate_text"]
                cur.execute(
                    """
                    INSERT INTO access_events (
                        entry_at, gate_id, vehicle_id, operator_id,
                        detected_plate_text, plate_text, raw_text, event_type, decision,
                        visitor_code, visitor_person_id, note,
                        detector_source, detector_score, ocr_confidence, bbox_json
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, 'exit', 'allowed',
                        %s, %s, %s,
                        'operator', 0, 0, '{}'::jsonb
                    )
                    RETURNING *
                    """,
                    (
                        exit_time,
                        gate_id,
                        session["vehicle_id"],
                        operator_id,
                        plate,
                        plate,
                        plate,
                        session["visitor_code"],
                        session["visitor_person_id"],
                        note,
                    ),
                )
                event = cur.fetchone()
                cur.execute(
                    """
                    UPDATE parking_sessions
                    SET
                        status = 'closed',
                        exited_at = %s,
                        exit_event_id = %s,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (exit_time, event["id"], session["id"]),
                )
                return event

    def get_stats(self) -> dict[str, int]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_reads,
                    COUNT(DISTINCT plate_text) AS unique_plates,
                    COUNT(*) FILTER (WHERE decision = 'allowed') AS allowed_reads,
                    COUNT(*) FILTER (WHERE decision = 'denied') AS denied_reads
                FROM access_events
                """
            )
            row = cur.fetchone()
            return {
                "total_reads": int(row["total_reads"] or 0),
                "unique_plates": int(row["unique_plates"] or 0),
                "allowed_reads": int(row["allowed_reads"] or 0),
                "denied_reads": int(row["denied_reads"] or 0),
            }

    def get_table_counts(self) -> dict[str, int]:
        tables = [
            "persons",
            "operators",
            "gates",
            "vehicle_makes",
            "vehicle_models",
            "vehicles",
            "vehicle_person_links",
            "vehicle_plate_aliases",
            "vehicle_access_rules",
            "access_events",
            "parking_sessions",
        ]
        counts: dict[str, int] = {}
        with self.conn.cursor() as cur:
            for table in tables:
                cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
                counts[table] = int(cur.fetchone()["count"] or 0)
        return counts

    def close(self) -> None:
        self.conn.close()
