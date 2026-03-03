# src/agents/triage.py
"""
Triage Agent - Document Classification Engine

This agent analyzes incoming documents and produces a DocumentProfile
that governs which extraction strategy will be used downstream.

Classification Dimensions:
1. Origin Type: native_digital | scanned_image | mixed | form_fillable
2. Layout Complexity: single_column | multi_column | table_heavy | figure_heavy | mixed
3. Language: detected language code + confidence
4. Domain Hint: financial | legal | technical | medical | general
5. Estimated Extraction Cost: fast_text_sufficient | needs_layout_model | needs_vision_model
"""

import pdfplumber
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional
from pydantic import ValidationError

from src.models.document_profile import DocumentProfile
from src.agents.config import Config


class TriageAgent:
    """
    Document classification agent that profiles documents before extraction.
    
    Uses empirical analysis (character density, image ratio, table detection)
    to classify documents and recommend extraction strategies.
    """
    
    # Domain keyword dictionaries for classification
    DOMAIN_KEYWORDS = {
        "financial": [
            "balance sheet", "income statement", "cash flow", "net profit",
            "revenue", "assets", "liabilities", "equity", "EBITDA", "ROI",
            "fiscal year", "quarterly", "audit", "compliance", "NBE directive",
            "liquidity ratio", "capital adequacy", "birr", "ETB", "bank"
        ],
        "legal": [
            "clause", "herein", "whereas", "plaintiff", "defendant", "court",
            "jurisdiction", "statute", "regulation", "compliance", "liability",
            "indemnification", "arbitration", "contract", "agreement"
        ],
        "technical": [
            "architecture", "implementation", "API", "endpoint", "database",
            "deployment", "infrastructure", "specification", "protocol",
            "integration", "system", "module", "component"
        ],
        "medical": [
            "diagnosis", "treatment", "patient", "clinical", "symptom",
            "medication", "dosage", "therapy", "prognosis", "healthcare"
        ]
    }
    
    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        """Initialize Triage Agent with configuration."""
        self.config = Config.load(config_path)
        self.triage_config = self.config.get('triage', {})
        self.sample_pages = self.triage_config.get('sample_pages', [1, 3, 5, 10, -1])
        
        # Thresholds from config
        classification = self.triage_config.get('classification', {})
        self.scanned_thresholds = classification.get('scanned_image', {})
        self.native_thresholds = classification.get('native_digital', {})
    
    def _compute_doc_id(self, filename: str, page_count: int) -> str:
        """Generate deterministic document ID."""
        data = f"{filename}|{page_count}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]
    
    def _sample_pages_to_analyze(self, total_pages: int) -> list[int]:
        """
        Determine which pages to sample for classification.
        
        Uses config-defined page indices, handling negative indices
        and ensuring we don't exceed document length.
        """
        pages_to_sample = []
        
        for idx in self.sample_pages:
            if idx == -1:
                # Last page
                page_num = total_pages
            elif idx > 0:
                page_num = min(idx, total_pages)
            else:
                continue
            
            if page_num not in pages_to_sample:
                pages_to_sample.append(page_num)
        
        return sorted(pages_to_sample)
    
    def _analyze_page(self, page: pdfplumber.page.Page) -> dict:
        """
        Extract empirical metrics from a single page.
        
        Returns metrics used for origin_type and layout_complexity classification.
        """
        page_area = page.width * page.height
        
        # Character analysis
        chars = page.chars if hasattr(page, 'chars') else []
        char_count = len(chars)
        char_density = (char_count / page_area) if page_area > 0 else 0
        
        # Image analysis
        images = page.images if hasattr(page, 'images') else []
        image_count = len(images)
        image_area = sum(
            img.get('width', 0) * img.get('height', 0) 
            for img in images
        )
        image_ratio = (image_area / page_area) if page_area > 0 else 0
        
        # Table detection
        try:
            tables = page.find_tables()
            table_count = len(tables)
        except Exception:
            table_count = 0
        
        # Font metadata analysis (indicates native digital)
        fonts_found = set()
        if chars:
            for char in chars[:100]:  # Sample first 100 chars
                font = char.get('fontname', '')
                if font:
                    fonts_found.add(font)
        
        # Compute bbox for first char (for provenance)
        first_char_bbox = None
        if chars:
            first_char = chars[0]
            first_char_bbox = {
                'x0': first_char.get('x0'),
                'top': first_char.get('top'),
                'x1': first_char.get('x1'),
                'bottom': first_char.get('bottom')
            }
        
        return {
            'page_num': page.page_number,
            'char_count': char_count,
            'char_density': char_density,
            'image_count': image_count,
            'image_ratio': image_ratio,
            'table_count': table_count,
            'fonts_found': list(fonts_found),
            'has_font_metadata': len(fonts_found) > 0,
            'first_char_bbox': first_char_bbox,
            'page_width': page.width,
            'page_height': page.height
        }
    
    def _compute_median(self, values: list[float]) -> float:
        """Compute median of a list of values."""
        if not values:
            return 0.0
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        
        if n % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        else:
            return sorted_vals[mid]
    
    def _classify_origin_type(self, metrics: list[dict]) -> tuple[Literal["native_digital", "scanned_image", "mixed", "form_fillable"], float]:
        """
        Classify document origin type based on empirical metrics.
        
        Returns: (origin_type, confidence)
        """
        char_densities = [m['char_density'] for m in metrics]
        image_ratios = [m['image_ratio'] for m in metrics]
        pages_with_text = sum(1 for m in metrics if m['char_count'] > 50)
        
        median_density = self._compute_median(char_densities)
        median_image_ratio = self._compute_median(image_ratios)
        
        # Get thresholds from config
        density_max_scanned = self.scanned_thresholds.get('median_char_density_max', 0.0005)
        density_min_native = self.native_thresholds.get('median_char_density_min', 0.001)
        image_ratio_min_scanned = self.scanned_thresholds.get('median_image_ratio_min', 0.7)
        image_ratio_max_native = self.native_thresholds.get('median_image_ratio_max', 0.4)
        min_pages_with_text = self.scanned_thresholds.get('min_pages_with_text', 2)
        
        # Classification logic
        if median_density < density_max_scanned and median_image_ratio > image_ratio_min_scanned:
            # Strong signals for scanned
            confidence = 0.95
            return "scanned_image", confidence
        
        elif median_density > density_min_native and median_image_ratio < image_ratio_max_native and pages_with_text >= min_pages_with_text:
            # Strong signals for native digital
            confidence = 0.90
            return "native_digital", confidence
        
        elif pages_with_text < min_pages_with_text:
            # Not enough text pages - likely scanned
            confidence = 0.85
            return "scanned_image", confidence
        
        else:
            # Mixed signals
            # Calculate confidence based on signal clarity
            density_signal = abs(median_density - density_max_scanned) / max(median_density, density_max_scanned)
            image_signal = abs(median_image_ratio - image_ratio_min_scanned)
            confidence = 0.5 + 0.25 * (density_signal + image_signal)
            
            return "mixed", min(confidence, 0.85)
    
    def _classify_layout_complexity(self, metrics: list[dict]) -> tuple[Literal["single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"], float]:
        """
        Classify layout complexity based on empirical metrics.
        
        Returns: (layout_complexity, confidence)
        """
        total_tables = sum(m['table_count'] for m in metrics)
        total_images = sum(m['image_count'] for m in metrics)
        avg_image_ratio = self._compute_median([m['image_ratio'] for m in metrics])
        
        # Simple heuristics for layout classification
        if total_tables >= 5:
            return "table_heavy", 0.85
        elif total_images >= 10 and avg_image_ratio > 0.3:
            return "figure_heavy", 0.80
        elif total_tables >= 2 or total_images >= 3:
            return "mixed", 0.75
        else:
            # Assume single column for simple docs
            return "single_column", 0.70
    
    def _classify_domain(self, pdf_path: Path) -> tuple[Literal["financial", "legal", "technical", "medical", "general"], list[str]]:
        """
        Classify document domain based on keyword matching.
        
        Returns: (domain_hint, keywords_found)
        """
        keywords_found = []
        domain_scores = {domain: 0 for domain in self.DOMAIN_KEYWORDS}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Sample first 5 pages for domain classification
                for page in pdf.pages[:5]:
                    text = page.extract_text() or ""
                    text_lower = text.lower()
                    
                    for domain, keywords in self.DOMAIN_KEYWORDS.items():
                        for keyword in keywords:
                            if keyword.lower() in text_lower:
                                domain_scores[domain] += 1
                                if keyword not in keywords_found:
                                    keywords_found.append(keyword)
        except Exception:
            pass
        
        # Determine winning domain
        if domain_scores:
            max_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[max_domain] >= 3:  # Minimum threshold
                return max_domain, keywords_found
        
        return "general", keywords_found
    
    def _estimate_extraction_cost(self, origin_type: str, layout_complexity: str) -> Literal["fast_text_sufficient", "needs_layout_model", "needs_vision_model"]:
        """
        Estimate extraction cost tier based on classification.
        """
        if origin_type == "scanned_image":
            return "needs_vision_model"
        elif layout_complexity in ["table_heavy", "multi_column", "mixed"]:
            return "needs_layout_model"
        else:
            return "fast_text_sufficient"
    
    def _recommend_strategy(self, origin_type: str, layout_complexity: str, cost_estimate: str) -> Literal["fast_text", "layout_aware", "vision_augmented"]:
        """
        Recommend extraction strategy based on profile.
        """
        if cost_estimate == "needs_vision_model":
            return "vision_augmented"
        elif cost_estimate == "needs_layout_model":
            return "layout_aware"
        else:
            return "fast_text"
    
    def profile_document(self, pdf_path: str) -> DocumentProfile:
        """
        Main entry point: Profile a document and return DocumentProfile.
        
        This method:
        1. Opens the PDF and samples pages
        2. Extracts empirical metrics from each sampled page
        3. Classifies origin_type, layout_complexity, domain
        4. Recommends extraction strategy
        5. Returns a validated DocumentProfile Pydantic model
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")
        
        # Open PDF and gather metadata
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            page_size = (pdf.pages[0].width, pdf.pages[0].height) if total_pages > 0 else (0, 0)
            
            # Determine which pages to sample
            pages_to_analyze = self._sample_pages_to_analyze(total_pages)
            
            # Extract metrics from each sampled page
            metrics = []
            for page_num in pages_to_analyze:
                page = pdf.pages[page_num - 1]  # Convert to 0-indexed
                page_metrics = self._analyze_page(page)
                metrics.append(page_metrics)
        
        # Classify document
        origin_type, origin_confidence = self._classify_origin_type(metrics)
        layout_complexity, layout_confidence = self._classify_layout_complexity(metrics)
        domain_hint, domain_keywords = self._classify_domain(pdf_path)
        
        # Estimate cost and recommend strategy
        cost_estimate = self._estimate_extraction_cost(origin_type, layout_complexity)
        recommended_strategy = self._recommend_strategy(origin_type, layout_complexity, cost_estimate)
        
        # Determine if escalation is likely needed
        requires_escalation = origin_type == "mixed" or layout_complexity in ["table_heavy", "multi_column"]
        escalation_reason = None
        if requires_escalation:
            if origin_type == "mixed":
                escalation_reason = f"Document has mixed content (image covers + text pages)"
            elif layout_complexity == "table_heavy":
                escalation_reason = f"Document contains {sum(m['table_count'] for m in metrics)} tables across sampled pages"
        
        # Compile metrics for profile
        profile_metrics = {
            'median_char_density': self._compute_median([m['char_density'] for m in metrics]),
            'median_image_ratio': self._compute_median([m['image_ratio'] for m in metrics]),
            'pages_with_text': sum(1 for m in metrics if m['char_count'] > 50),
            'table_count': sum(m['table_count'] for m in metrics),
            'char_density_stddev': self._compute_stddev([m['char_density'] for m in metrics]),
            'sampled_pages': pages_to_analyze,
            'total_pages': total_pages,
            'page_size': page_size
        }
        
        # Create DocumentProfile
        profile = DocumentProfile(
            doc_id=self._compute_doc_id(pdf_path.name, total_pages),
            filename=pdf_path.name,
            origin_type=origin_type,
            origin_confidence=origin_confidence,
            layout_complexity=layout_complexity,
            layout_confidence=layout_confidence,
            language="en",  # Default, can be enhanced with langdetect
            language_confidence=0.95,
            page_count=total_pages,
            domain_hint=domain_hint,
            domain_keywords_found=domain_keywords,
            metrics=profile_metrics,
            recommended_strategy=recommended_strategy,
            estimated_extraction_cost=cost_estimate,
            requires_escalation=requires_escalation,
            escalation_reason=escalation_reason
        )
        
        return profile
    
    def _compute_stddev(self, values: list[float]) -> float:
        """Compute standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def save_profile(self, profile: DocumentProfile, output_dir: str = ".refinery/profiles") -> Path:
        """Save DocumentProfile to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use doc_id for filename
        profile_file = output_path / f"{profile.doc_id}.json"
        
        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(profile.model_dump(mode='json'), f, indent=2, default=str)
        
        return profile_file


def main():
    """CLI entry point for triaging documents."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.triage <pdf_path> [output_dir]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else ".refinery/profiles"
    
    agent = TriageAgent()
    
    print(f"🔍 Profiling document: {pdf_path}")
    profile = agent.profile_document(pdf_path)
    
    print(f"\n📄 Document Profile:")
    print(f"  Doc ID: {profile.doc_id}")
    print(f"  Filename: {profile.filename}")
    print(f"  Origin Type: {profile.origin_type} (confidence: {profile.origin_confidence:.2f})")
    print(f"  Layout Complexity: {profile.layout_complexity} (confidence: {profile.layout_confidence:.2f})")
    print(f"  Domain: {profile.domain_hint}")
    print(f"  Pages: {profile.page_count}")
    print(f"  Recommended Strategy: {profile.recommended_strategy}")
    print(f"  Estimated Cost: {profile.estimated_extraction_cost}")
    print(f"  Requires Escalation: {profile.requires_escalation}")
    if profile.escalation_reason:
        print(f"  Escalation Reason: {profile.escalation_reason}")
    
    # Save profile
    profile_path = agent.save_profile(profile, output_dir)
    print(f"\n💾 Profile saved to: {profile_path}")
    
    return profile


if __name__ == "__main__":
    main()