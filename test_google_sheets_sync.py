"""
Phase 7 Test Suite: Google Sheets Synchronization Engine
Tests:
1. Configuration detection and credential handling.
2. Header construction: 'Student ID', 'Student Name', and 'DD-MM-YYYY' date columns.
3. Auto-creation of date column when missing.
4. Resubmission update logic (avoiding duplicate date columns).
5. Mandatory Safety Rule: Never downgrade an existing 'P' to 'A'.
6. Upgrading 'A' to 'P' upon re-scan/correction.
7. Batch update payload generation.
"""

import sys
import os
from datetime import date
from unittest.mock import MagicMock, patch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app.google_sheets.sync import GoogleSheetsSync

def run_tests():
    print("=" * 70)
    print("PHASE 7: GOOGLE SHEETS SYNCHRONIZATION TEST SUITE")
    print("=" * 70)

    # 1. Test Configuration Detection
    print("\n[Step 1/5] Verifying Configuration Detection...")
    unconfigured_sync = GoogleSheetsSync(service_account_source=None, sheet_id=None)
    assert unconfigured_sync.is_configured() is False
    success, msg = unconfigured_sync.sync_attendance(date.today(), {}, {})
    assert success is False
    assert "not configured" in msg
    print("  [PASS] Unconfigured environment handled gracefully without crashing")

    # 2. Test Sheet Structure & Mock Synchronization Pipeline
    print("\n[Step 2/5] Testing Sheet Header & Row Synchronization...")
    mock_worksheet = MagicMock()

    # Initial state: completely empty sheet
    mock_worksheet.get_all_values.return_value = []

    mock_spreadsheet = MagicMock()
    mock_spreadsheet.sheet1 = mock_worksheet

    mock_client = MagicMock()
    mock_client.open_by_key.return_value = mock_spreadsheet

    sync_service = GoogleSheetsSync(
        service_account_source="dummy_creds.json",
        sheet_id="mock_sheet_id_123"
    )

    with patch.object(sync_service, "get_client", return_value=mock_client), \
         patch.object(sync_service, "is_configured", return_value=True):

        attendance_date_1 = date(2026, 11, 4)
        att_map = {
            "STU001": "P",
            "STU002": "A"
        }
        names = {
            "STU001": "Alice Smith",
            "STU002": "Bob Jones"
        }

        success, msg = sync_service.sync_attendance(attendance_date_1, att_map, names)
        assert success is True
        print(f"  [PASS] Initial sync passed: {msg}")
        mock_worksheet.append_row.assert_called_with(["Student ID", "Student Name", "04-11-2026"])
        assert mock_worksheet.append_rows.called

    # 3. Test Non-Duplication of Date Column on Resubmission
    print("\n[Step 3/5] Testing Resubmission on Existing Date Column...")
    # Sheet now has data from 04-11-2026
    current_sheet_state = [
        ["Student ID", "Student Name", "04-11-2026"],
        ["STU001", "Alice Smith", "P"],
        ["STU002", "Bob Jones", "A"]
    ]
    mock_worksheet.get_all_values.return_value = [row[:] for row in current_sheet_state]
    mock_worksheet.reset_mock()

    with patch.object(sync_service, "get_client", return_value=mock_client), \
         patch.object(sync_service, "is_configured", return_value=True):

        # Re-submit for 04-11-2026
        success, msg = sync_service.sync_attendance(attendance_date_1, att_map, names)
        assert success is True
        # Date column should NOT be appended again
        mock_worksheet.append_row.assert_not_called()
        print("  [PASS] Re-submitting on the same date reused existing date column without duplicating")

    # 4. Mandatory Safety Rule: Never Downgrade 'P' to 'A'
    print("\n[Step 4/5] Testing Mandatory Rule: NEVER Downgrade 'P' to 'A'...")
    # Current sheet state: STU001 is 'P'
    mock_worksheet.get_all_values.return_value = [
        ["Student ID", "Student Name", "04-11-2026"],
        ["STU001", "Alice Smith", "P"],
        ["STU002", "Bob Jones", "A"]
    ]
    mock_worksheet.reset_mock()

    with patch.object(sync_service, "get_client", return_value=mock_client), \
         patch.object(sync_service, "is_configured", return_value=True):

        # Accidental/uncertain re-scan marking STU001 as 'A'
        downgrade_attempt_map = {
            "STU001": "A",  # Attempted downgrade!
            "STU002": "P"   # Legitimate upgrade from 'A' to 'P'
        }

        success, msg = sync_service.sync_attendance(attendance_date_1, downgrade_attempt_map, names)
        assert success is True

        # Verify batch updates:
        # STU001 should NOT have been changed to 'A'
        # STU002 SHOULD have been changed from 'A' to 'P'
        calls = mock_worksheet.batch_update.call_args_list
        assert len(calls) == 1
        batch_payload = calls[0][0][0]

        # Check payload
        updated_cells = {item["range"]: item["values"][0][0] for item in batch_payload}
        # STU002 is on row 3, column C ('04-11-2026') -> C3 = 'P'
        assert "C3" in updated_cells and updated_cells["C3"] == "P"
        # STU001 (row 2, C2) was already 'P', so it must NOT be downgraded to 'A'
        assert "C2" not in updated_cells or updated_cells["C2"] == "P"
        print("  [PASS] Downgrade blocked: Existing 'P' preserved against accidental 'A' override")
        print("  [PASS] Upgrade permitted: 'A' successfully upgraded to 'P' for corrected student")

    # 5. Adding New Date Column for Next Day (05-11-2026)
    print("\n[Step 5/5] Testing Dynamic Multi-Date Expansion (05-11-2026)...")
    mock_worksheet.get_all_values.return_value = [
        ["Student ID", "Student Name", "04-11-2026"],
        ["STU001", "Alice Smith", "P"],
        ["STU002", "Bob Jones", "P"]
    ]
    mock_worksheet.reset_mock()

    with patch.object(sync_service, "get_client", return_value=mock_client), \
         patch.object(sync_service, "is_configured", return_value=True):

        attendance_date_2 = date(2026, 11, 5)
        att_map_day2 = {
            "STU001": "P",
            "STU002": "A"
        }

        success, msg = sync_service.sync_attendance(attendance_date_2, att_map_day2, names)
        assert success is True
        mock_worksheet.update_cell.assert_called_with(1, 4, "05-11-2026")
        print("  [PASS] New date column '05-11-2026' automatically appended as Column D")

    print("\n" + "=" * 70)
    print("ALL PHASE 7 GOOGLE SHEETS TESTS PASSED (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()

