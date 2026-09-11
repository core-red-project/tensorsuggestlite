"""Dataset profiler for detecting anomalies, imbalance, and contradictions."""

from collections import Counter, defaultdict

from tensorsuggestlite.core.entities import DatasetProfileReport, TrainingDataset


class DatasetProfiler:
    """Profiles a TrainingDataset to detect data quality issues before training."""

    @staticmethod
    def profile(dataset: TrainingDataset) -> DatasetProfileReport:
        """Run full diagnostic analysis on a dataset."""
        total = len(dataset.samples)
        if total == 0:
            return DatasetProfileReport(
                total_samples=0,
                unique_texts=0,
                category_counts={},
                imbalance_ratio=1.0,
                imbalance_warning=None,
                duplicate_samples=[],
                contradictory_samples=[],
            )

        cat_counts = Counter(s.category for s in dataset.samples)
        min_count = min(cat_counts.values()) if cat_counts else 1
        max_count = max(cat_counts.values()) if cat_counts else 1
        ratio = max_count / max(min_count, 1)

        warning: str | None = None
        if total > 5:
            if ratio > 4.0:
                warning = f"Critical class imbalance: ratio {ratio:.1f}:1 between largest and smallest class."
            elif ratio > 2.5:
                warning = f"Moderate class imbalance: ratio {ratio:.1f}:1."
            elif min_count < 3:
                warning = f"Under-represented class detected (minimum count: {min_count} samples)."

        # Duplicates and contradictions
        text_to_categories: defaultdict[str, set[str]] = defaultdict(set)
        text_cat_counts: Counter[tuple[str, str]] = Counter()

        for s in dataset.samples:
            normalized_text = s.text.strip().lower()
            text_to_categories[normalized_text].add(s.category)
            text_cat_counts[(s.text.strip(), s.category)] += 1

        contradictions: list[tuple[str, list[str]]] = []
        for text, cats in text_to_categories.items():
            if len(cats) > 1:
                contradictions.append((text, sorted(list(cats))))

        duplicates: list[tuple[str, str, int]] = []
        for (text, cat), count in text_cat_counts.items():
            if count > 1:
                duplicates.append((text, cat, count))

        return DatasetProfileReport(
            total_samples=total,
            unique_texts=len(text_to_categories),
            category_counts=dict(cat_counts),
            imbalance_ratio=ratio,
            imbalance_warning=warning,
            duplicate_samples=duplicates,
            contradictory_samples=contradictions,
        )
