from policy_lens import check_eligibility, chunk_text

def test_pm_kisan_happy_path():
    result = check_eligibility("PM-KISAN", {"land_hectares": 1, "documents": ["Aadhaar", "Bank account", "Land record"]})
    assert result["eligible"] is True

def test_pm_kisan_lists_missing_documents():
    result = check_eligibility("PM-KISAN", {"land_hectares": 1, "documents": ["Aadhaar"]})
    assert result["eligible"] is False
    assert "Bank account" in result["missing_documents"]

def test_empty_text_does_not_create_chunks():
    assert chunk_text("") == []
