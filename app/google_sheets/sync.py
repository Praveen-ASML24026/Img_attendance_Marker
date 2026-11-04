import os
import json
from datetime import datetime, date
from typing import Tuple, Dict, List, Optional, Any

import gspread
from google.oauth2.service_account import Credentials
from google.auth.exceptions import GoogleAuthError

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


class GoogleSheetsSync:
    """
    Synchronizes teacher-confirmed attendance sessions with Google Sheets.
    Implements:
    - Service account authentication (from file path or JSON string in .env)
    - Auto-creation and detection of date columns (format DD-MM-YYYY)
    - Non-downgrade rule: Never downgrades an existing 'P' to 'A' upon resubmission
    - Batch update execution to avoid API rate limits
    - Graceful error reporting when credentials are not configured
    """

    def __init__(
        self,
        service_account_source: Optional[str] = None,
        sheet_id: Optional[str] = None
    ):
        self.service_account_source = service_account_source or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        self.sheet_id = sheet_id or os.getenv("GOOGLE_SHEET_ID")
        self._client: Optional[gspread.Client] = None

    def is_configured(self) -> bool:
        """Returns True if both credentials and sheet ID are configured."""
        return bool(self.service_account_source and self.sheet_id)

    def _get_credentials(self) -> Optional[Credentials]:
        """Loads Google Service Account credentials from JSON file or JSON string."""
        if not self.service_account_source:
            return None

        try:
            # Check if source is a file path
            if os.path.isfile(self.service_account_source):
                return Credentials.from_service_account_file(
                    self.service_account_source,
                    scopes=SCOPES
                )

            # Check if source is a raw JSON string
            if self.service_account_source.strip().startswith("{"):
                info = json.loads(self.service_account_source)
                return Credentials.from_service_account_info(
                    info,
                    scopes=SCOPES
                )

            # Check relative to project instance folder
            possible_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..",
                self.service_account_source
            )
            if os.path.isfile(possible_path):
                return Credentials.from_service_account_file(
                    possible_path,
                    scopes=SCOPES
                )

        except Exception as e:
            return None

        return None

    def get_client(self) -> Optional[gspread.Client]:
        """Lazily authenticates and returns the gspread client."""
        if self._client is None:
            creds = self._get_credentials()
            if creds:
                self._client = gspread.authorize(creds)
        return self._client

    def sync_attendance(
        self,
        session_date: date,
        attendance_map: Dict[str, str],
        student_names: Dict[str, str]
    ) -> Tuple[bool, str]:
        """
        Synchronizes an attendance session with Google Sheets.
        
        Args:
            session_date: The date of the session (e.g. 2026-11-04)
            attendance_map: Dict mapping student_id -> status ('P' or 'A')
            student_names: Dict mapping student_id -> full name
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.is_configured():
            return False, "Google Sheets credentials or Sheet ID are not configured in .env."

        client = self.get_client()
        if not client:
            return False, "Failed to authenticate Google Service Account. Please verify credentials."

        try:
            spreadsheet = client.open_by_key(self.sheet_id)
            worksheet = spreadsheet.sheet1
        except Exception as e:
            return False, f"Could not access Google Sheet (ID: {self.sheet_id}): {str(e)}"

        try:
            date_str = session_date.strftime("%d-%m-%Y")
            all_values = worksheet.get_all_values()

            # 1. Initialize or inspect Header Row
            if not all_values:
                headers = ["Student ID", "Student Name", date_str]
                all_values = [headers]
                worksheet.append_row(headers)
                date_col_idx = 2  # 0-indexed column index
            else:
                headers = all_values[0]
                # Ensure permanent columns exist
                if len(headers) < 2 or headers[0] != "Student ID":
                    headers = ["Student ID", "Student Name"] + [h for h in headers if h not in ("Student ID", "Student Name")]
                    all_values[0] = headers

                # Find or append date column
                if date_str in headers:
                    date_col_idx = headers.index(date_str)
                else:
                    date_col_idx = len(headers)
                    headers.append(date_str)
                    # Update header row in sheet
                    worksheet.update_cell(1, date_col_idx + 1, date_str)

            # 2. Build or update student row mapping
            student_row_map = {}
            for row_idx in range(1, len(all_values)):
                row = all_values[row_idx]
                if row and row[0]:
                    student_row_map[row[0]] = row_idx

            # 3. Synchronize rows in memory
            rows_to_append = []
            updates_to_make = []

            for student_id, new_status in attendance_map.items():
                name = student_names.get(student_id, f"Student {student_id}")

                if student_id in student_row_map:
                    row_idx = student_row_map[student_id]
                    current_row = all_values[row_idx]

                    # Pad row if shorter than headers
                    while len(current_row) < len(headers):
                        current_row.append("")

                    existing_status = current_row[date_col_idx].strip() if date_col_idx < len(current_row) else ""

                    # RULE: Never downgrade an existing 'P' to 'A'
                    final_status = "P" if (existing_status == "P" or new_status == "P") else new_status

                    if existing_status != final_status:
                        current_row[date_col_idx] = final_status
                        # 1-indexed cell coordinates for gspread
                        updates_to_make.append({
                            "range": gspread.utils.rowcol_to_a1(row_idx + 1, date_col_idx + 1),
                            "values": [[final_status]]
                        })
                else:
                    # New student not yet in spreadsheet
                    new_row = [student_id, name] + [""] * (len(headers) - 2)
                    new_row[date_col_idx] = new_status
                    all_values.append(new_row)
                    rows_to_append.append(new_row)

            # 4. Execute updates efficiently
            if rows_to_append:
                worksheet.append_rows(rows_to_append)

            if updates_to_make:
                worksheet.batch_update(updates_to_make)

            present_count = sum(1 for s in attendance_map.values() if s == "P")
            return True, f"Successfully synced {len(attendance_map)} students to Google Sheets for {date_str} ({present_count} Present)."

        except Exception as e:
            return False, f"Error synchronizing with Google Sheets: {str(e)}"

