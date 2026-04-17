#!/usr/bin/env python3
"""Step 18 Verification — Template Engine Tests"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DRY_RUN"] = "true"

def _make_sections():
    return [
        {"section_id": "1.0", "section_title": "Company Overview",
         "content": "Acme SDS Solutions is a leading provider of Safety Data Sheet management software, serving over 500 clients across manufacturing, construction, oil and gas, and pharmaceutical industries. Founded with a mission to simplify chemical safety compliance, our platform provides comprehensive SDS management capabilities trusted by organizations of all sizes.",
         "confidence": 0.92, "sources_used": ["profile.pdf"], "model_used": "qwen3.5-plus", "token_count": 200},
        {"section_id": "2.0", "section_title": "Technical Capabilities",
         "content": "Our platform provides cloud-based SDS management including GHS classification, chemical inventory tracking across multiple facilities, automated regulatory reporting (OSHA Tier II, EPCRA/CERCLA), and mobile access via QR codes and native apps. We support OSHA HCS, WHMIS, CLP/REACH, and over 140 global regulatory jurisdictions.",
         "confidence": 0.88, "sources_used": ["caps.pdf"], "model_used": "qwen3.5-plus", "token_count": 250},
        {"section_id": "3.0", "section_title": "Compliance & Certifications",
         "content": "Acme SDS Solutions holds ISO 27001 certification for information security and maintains SOC 2 Type II compliance, verified annually by independent auditors. Our data centres operate in AWS US-East and AWS EU-West regions, providing 99.95% uptime SLA with full data redundancy.",
         "confidence": 0.85, "sources_used": ["certs.pdf"], "model_used": "qwen3-max", "token_count": 300},
        {"section_id": "4.0", "section_title": "Implementation Approach",
         "content": "Our standard implementation follows a proven 12-week methodology: Week 1-2 Discovery and requirements. Week 3-4 System configuration and data migration planning. Week 5-8 Data migration and integration. Week 9-10 User acceptance testing. Week 11-12 Training and go-live. Post-launch support for 90 days.",
         "confidence": 0.90, "sources_used": ["impl.pdf"], "model_used": "qwen3.5-plus", "token_count": 220},
    ]

def test_1_standard_assembly():
    from src.assembly.template_engine import TemplateEngine
    engine = TemplateEngine()
    doc = engine.assemble(_make_sections(), "SDS Platform for EPA Region 5", "SAM-2026-001", "standard", "June 15, 2026")
    assert "# SDS Platform for EPA Region 5" in doc
    assert "Acme SDS Solutions" in doc
    assert "SAM-2026-001" in doc
    assert "Table of Contents" in doc
    assert "## 1.0 Company Overview" in doc
    assert "## 4.0 Implementation Approach" in doc
    assert "Tender Agent" in doc
    print(f"  ✅ Test 1 passed: Standard template assembled ({len(doc.split())} words)")

def test_2_simple_no_cover():
    from src.assembly.template_engine import TemplateEngine
    engine = TemplateEngine()
    doc = engine.assemble(_make_sections(), "Quick Proposal", "QP-001", "simple")
    assert "# Quick Proposal" not in doc
    assert "Table of Contents" not in doc
    assert "## 1.0 Company Overview" in doc
    print("  ✅ Test 2 passed: Simple template has no cover page or TOC")

def test_3_quality_passes():
    from src.assembly.template_engine import TemplateEngine
    engine = TemplateEngine()
    sections = _make_sections()
    doc = engine.assemble(sections, "Test", "T-001")
    qc = engine.quality_check(doc, sections=sections)
    assert qc.passed is True, f"Should pass, issues: {qc.issues}"
    assert qc.stats["word_count"] > 100
    assert qc.stats["section_count"] == 4
    print(f"  ✅ Test 3 passed: Quality check passes (words={qc.stats['word_count']}, pages~{qc.stats['page_estimate']})")

def test_4_quality_catches_problems():
    from src.assembly.template_engine import TemplateEngine
    engine = TemplateEngine()
    bad = [{"section_id": "1.0", "section_title": "Overview",
            "content": "Good services. [INFORMATION NEEDED: client count] [INFORMATION NEEDED: revenue]",
            "confidence": 0.5, "sources_used": [], "model_used": "test", "token_count": 50}]
    doc = engine.assemble(bad, "Test", "T-002")
    qc = engine.quality_check(doc, sections=bad)
    assert qc.passed is False, "Should fail (placeholders)"
    assert any("placeholder" in i.lower() for i in qc.issues)
    assert any("confidence" in w.lower() for w in qc.warnings)
    # Page limit test
    huge = [{"section_id": f"{i}.0", "section_title": f"Sec {i}",
             "content": "Detailed section content here. " * 300,
             "confidence": 0.9, "sources_used": [], "model_used": "t", "token_count": 1000}
            for i in range(1, 6)]
    huge_doc = engine.assemble(huge, "Huge", "T-003", "simple")
    huge_qc = engine.quality_check(huge_doc, sections=huge, template_name="simple")
    assert huge_qc.passed is False, "Should fail page limit"
    assert any("page" in i.lower() for i in huge_qc.issues)
    print("  ✅ Test 4 passed: Quality check catches placeholders and page limits")

def test_5_save_document():
    from src.assembly.template_engine import TemplateEngine
    with tempfile.TemporaryDirectory(prefix="tender_test_") as tmpdir:
        engine = TemplateEngine(output_dir=tmpdir)
        doc = engine.assemble(_make_sections(), "Save Test", "T-005")
        path1 = engine.save(doc)
        assert Path(path1).exists()
        assert Path(path1).read_text(encoding="utf-8") == doc
        explicit = str(Path(tmpdir) / "my_tender.md")
        path2 = engine.save(doc, filename=explicit)
        assert Path(path2).exists()
        assert path2 == explicit
        print(f"  ✅ Test 5 passed: Documents save correctly (auto={Path(path1).name}, explicit={Path(path2).name})")

def main():
    print("\n" + "=" * 60)
    print("  Step 18 Verification: Template Engine")
    print("=" * 60 + "\n")
    tests = [
        ("Test 1: Standard template assembly", test_1_standard_assembly),
        ("Test 2: Simple template (no cover)", test_2_simple_no_cover),
        ("Test 3: Quality check passes", test_3_quality_passes),
        ("Test 4: Quality check catches problems", test_4_quality_catches_problems),
        ("Test 5: Save document", test_5_save_document),
    ]
    passed = failed = 0
    for name, fn in tests:
        try: fn(); passed += 1
        except AssertionError as e: print(f"  ❌ {name} FAILED: {e}"); failed += 1
        except Exception as e: print(f"  ❌ {name} ERROR: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{'=' * 60}\n  Results: {passed} passed, {failed} failed\n{'=' * 60}\n")
    if failed: sys.exit(1)
    else:
        print("  🎉 All tests passed! Step 18 is complete.")
        print("  Next: git add -A && git commit -m 'Step 18: Template engine'\n")

if __name__ == "__main__":
    main()