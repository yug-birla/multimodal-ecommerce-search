import os
import re
import json
import math
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import redis
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels


# =====================================================================
# AMAZON 4-CATEGORY CONFIG
# Your raw files:
#   meta_Amazon_Fashion.jsonl.gz
#   meta_Beauty_and_Personal_Care.jsonl.gz
#   meta_Home_and_Kitchen.jsonl.gz
#   meta_Sports_and_Outdoors.jsonl.gz
#
# Keep the dictionary KEYS close to what you store in Qdrant payload.
# If your payload stores exact file stems, these keys are already correct.
# =====================================================================

CATEGORY_ALIASES: Dict[str, List[str]] = {
    "Amazon_Fashion": [
        "amazon fashion", "fashion", "clothing", "clothes", "apparel", "wear", "outfit",
        "men", "mens", "women", "womens", "girl", "boy", "unisex",
        "shirt", "tshirt", "t-shirt", "tee", "top", "crop top", "blouse", "kurti", "kurta",
        "dress", "saree", "jeans", "pant", "pants", "trouser", "trousers", "shorts",
        "jacket", "hoodie", "sweater", "coat", "skirt", "leggings",
        "shoe", "shoes", "sneaker", "sneakers", "slipper", "sandals", "heel", "heels", "boot", "boots",
        "watch", "bag", "handbag", "wallet", "belt", "sunglasses", "cap", "hat", "jewelry", "jewellery",
    ],
    "Beauty_and_Personal_Care": [
        "beauty and personal care", "beauty", "personal care", "skin", "skincare", "skin care",
        "hair", "haircare", "hair care", "makeup", "cosmetic", "cosmetics",
        "cream", "moisturizer", "moisturiser", "serum", "facewash", "face wash", "cleanser",
        "sunscreen", "spf", "lotion", "toner", "lipstick", "lip balm", "mascara", "eyeliner",
        "foundation", "concealer", "compact", "blush", "nail", "shampoo", "conditioner",
        "hair oil", "perfume", "deodorant", "body wash", "soap", "razor", "trimmer",
    ],
    "Home_and_Kitchen": [
        "home and kitchen", "home", "kitchen", "house", "decor", "decoration", "furniture",
        "cookware", "utensil", "utensils", "pan", "pot", "kadai", "tawa", "knife", "knives",
        "mixer", "grinder", "bottle", "flask", "mug", "cup", "plate", "bowl", "spoon", "container",
        "storage", "organizer", "organiser", "rack", "shelf", "bedsheet", "bed sheet", "pillow",
        "blanket", "curtain", "lamp", "light", "mat", "carpet", "towel", "cleaner", "bathroom",
    ],
    "Sports_and_Outdoors": [
        "sports and outdoors", "sports", "outdoors", "outdoor", "fitness", "gym", "workout", "exercise",
        "yoga", "running", "cycling", "cricket", "football", "badminton", "tennis", "basketball",
        "swimming", "camping", "hiking", "trekking", "travel", "backpack", "rucksack", "duffel",
        "bottle", "gloves", "helmet", "ball", "bat", "racket", "racquet", "dumbbell", "resistance band",
        "mat", "tent", "fishing", "sportswear", "tracksuit", "track pant", "jersey",
    ],
}

# Values that may appear in payload['category'], payload['main_category'], or source file name.
# This makes category matching robust even if you stored category in different formats.
CATEGORY_VALUE_VARIANTS: Dict[str, List[str]] = {
    "Amazon_Fashion": [
        "Amazon_Fashion", "Amazon Fashion", "amazon fashion", "meta_Amazon_Fashion.jsonl.gz",
        "meta_Amazon_Fashion", "fashion", "AMAZON_FASHION",
    ],
    "Beauty_and_Personal_Care": [
        "Beauty_and_Personal_Care", "Beauty and Personal Care", "beauty and personal care",
        "meta_Beauty_and_Personal_Care.jsonl.gz", "meta_Beauty_and_Personal_Care",
        "beauty", "personal care", "BEAUTY_AND_PERSONAL_CARE",
    ],
    "Home_and_Kitchen": [
        "Home_and_Kitchen", "Home and Kitchen", "home and kitchen",
        "meta_Home_and_Kitchen.jsonl.gz", "meta_Home_and_Kitchen", "home", "kitchen",
        "HOME_AND_KITCHEN",
    ],
    "Sports_and_Outdoors": [
        "Sports_and_Outdoors", "Sports and Outdoors", "sports and outdoors",
        "meta_Sports_and_Outdoors.jsonl.gz", "meta_Sports_and_Outdoors", "sports", "outdoors",
        "SPORTS_AND_OUTDOORS",
    ],
}

# No external NLP package. Just hand-written e-commerce synonym expansion.
SYNONYMS: Dict[str, List[str]] = {
    # Fashion
    "tshirt": ["t-shirt", "tee", "shirt"],
    "t-shirt": ["tshirt", "tee", "shirt"],
    "tee": ["tshirt", "t-shirt", "shirt"],
    "sneaker": ["sneakers", "shoe", "shoes", "casual shoe"],
    "sneakers": ["sneaker", "shoe", "shoes", "casual shoe"],
    "trainer": ["sneaker", "sports shoe", "running shoe"],
    "wallet": ["purse", "card holder"],
    "handbag": ["bag", "purse"],
    "jewelry": ["jewellery", "accessory"],
    "jewellery": ["jewelry", "accessory"],

    # Beauty
    "moisturiser": ["moisturizer", "cream", "lotion"],
    "moisturizer": ["moisturiser", "cream", "lotion"],
    "facewash": ["face wash", "cleanser"],
    "cleanser": ["facewash", "face wash"],
    "spf": ["sunscreen", "sunblock"],
    "sunscreen": ["spf", "sunblock"],
    "perfume": ["fragrance", "deodorant"],
    "deodorant": ["deo", "perfume", "fragrance"],
    "shampoo": ["hair wash", "haircare"],

    # Home/Kitchen
    "organizer": ["organiser", "storage", "rack"],
    "organiser": ["organizer", "storage", "rack"],
    "flask": ["bottle", "water bottle"],
    "bottle": ["flask", "water bottle"],
    "bedsheet": ["bed sheet", "bedding"],
    "pan": ["cookware", "tawa", "kadai"],
    "container": ["storage", "box", "jar"],

    # Sports/Outdoors
    "gym": ["fitness", "workout", "exercise"],
    "workout": ["gym", "fitness", "exercise"],
    "yoga": ["fitness", "exercise", "mat"],
    "backpack": ["rucksack", "bag", "travel bag"],
    "rucksack": ["backpack", "travel bag"],
    "racquet": ["racket"],
    "racket": ["racquet"],

    # Shopping intent
    "cheap": ["budget", "affordable", "low price", "value for money"],
    "affordable": ["budget", "cheap", "low price", "value for money"],
    "premium": ["high quality", "branded"],
    "durable": ["strong", "long lasting"],
    "lightweight": ["light weight", "portable"],

    # Common spelling/attribute
    "black": ["dark"],
    "white": ["light"],
    "grey": ["gray"],
    "gray": ["grey"],
}

STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "with", "and", "or", "by",
    "is", "are", "show", "find", "search", "product", "products", "item", "items",
    "buy", "best", "good", "nice", "high", "quality", "online", "amazon",
    "under", "below", "above", "over", "less", "than", "between", "from", "upto", "up", "max", "min",
    "price", "priced", "range", "rs", "rupees", "inr", "dollar", "dollars",
}

COLORS = {
    "black", "white", "red", "blue", "green", "yellow", "pink", "purple", "orange",
    "brown", "grey", "gray", "gold", "silver", "beige", "cream", "maroon", "navy",
}

GENDER_TERMS = {"men", "mens", "male", "women", "womens", "female", "boys", "girls", "unisex"}


@dataclass
class ParsedQuery:
    raw: str
    clean: str
    tokens: List[str]
    expanded_tokens: List[str]
    category_intent: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    colors: List[str] = field(default_factory=list)
    gender_terms: List[str] = field(default_factory=list)
    must_terms: List[str] = field(default_factory=list)


class SearchService:
    """
    Semantic hybrid multimodal e-commerce search for your 4 Amazon categories.

    What it does WITHOUT retraining:
    - CLIP semantic retrieval from Qdrant.
    - Query parsing: category, price, color, gender/use-case terms.
    - Synonym expansion for e-commerce words.
    - Multi-prompt CLIP text embedding.
    - Local reranking using semantic + lexical + category + price + rating/review signals.
    - Duplicate control so top results are not 6 copies of the same item.
    - Optional strict Qdrant payload filtering, disabled by default for low-data projects.
    """

    CACHE_VERSION = "amazon_4cat_semantic_v1"

    def __init__(
        self,
        collection_name: str = "products",
        model_id: str = "openai/clip-vit-base-patch32",
        enable_payload_filter: bool = False,
        redis_ttl_seconds: int = 1800,
    ):
        print("Initializing Amazon 4-Category Semantic Hybrid Search Service...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.collection_name = collection_name
        self.enable_payload_filter = enable_payload_filter
        self.redis_ttl_seconds = redis_ttl_seconds

        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60.0,
        )

        self.redis_client = self._connect_redis_safely()

        print(f"Loading CLIP Multimodal Encoder ({self.device.upper()} Mode)...")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    # =================================================================
    # PUBLIC SEARCH METHODS
    # =================================================================

    def search(
        self,
        text_query: str,
        top_k: int = 6,
        fetch_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Text-to-product search."""
        parsed = self._parse_query(text_query)
        cache_key = self._cache_key("text", parsed.raw, top_k, fetch_k)

        if use_cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                print(f"⚡ CACHE HIT: {text_query}")
                return cached

        print(f"🧠 TEXT SEARCH: '{text_query}'")
        print(
            f"   Parsed => category={parsed.category_intent}, "
            f"price=({parsed.min_price}, {parsed.max_price}), "
            f"tokens={parsed.must_terms}"
        )

        vector = self._text_embedding(parsed)
        results = self._execute_qdrant_query(
            vector=vector,
            top_k=top_k,
            fetch_k=fetch_k,
            parsed_query=parsed,
        )

        if use_cache:
            self._write_cache(cache_key, results)
        return results

    def search_image(
        self,
        image_file,
        top_k: int = 6,
        fetch_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Image-to-product search."""
        print("📸 IMAGE SEARCH: processing uploaded product image...")
        vector = self._image_embedding(image_file)
        return self._execute_qdrant_query(
            vector=vector,
            top_k=top_k,
            fetch_k=fetch_k,
            parsed_query=None,
        )

    def search_composed(
        self,
        image_file,
        text_query: str,
        top_k: int = 6,
        fetch_k: Optional[int] = None,
        image_weight: float = 0.70,
        text_weight: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """
        Image + text constrained search.
        Example: upload shoe photo + text_query='black running shoes under 2000'.
        """
        parsed = self._parse_query(text_query)
        print(f"🔮 COMPOSED SEARCH: image + '{text_query}'")

        img_vec = torch.tensor(self._image_embedding(image_file), dtype=torch.float32, device=self.device)
        txt_vec = torch.tensor(self._text_embedding(parsed), dtype=torch.float32, device=self.device)

        image_weight = self._clip01(image_weight)
        text_weight = self._clip01(text_weight)
        if image_weight + text_weight == 0:
            image_weight, text_weight = 0.70, 0.30
        total = image_weight + text_weight
        image_weight, text_weight = image_weight / total, text_weight / total

        combined = image_weight * img_vec + text_weight * txt_vec
        combined = combined / combined.norm(p=2)
        vector = combined.detach().cpu().numpy().tolist()

        return self._execute_qdrant_query(
            vector=vector,
            top_k=top_k,
            fetch_k=fetch_k,
            parsed_query=parsed,
        )

    def ensure_payload_indexes(self) -> None:
        """
        Optional utility. Run once if you later enable strict payload filtering.
        It is safe to ignore errors because index creation may vary by Qdrant version.
        """
        index_plan = [
            ("category", qmodels.PayloadSchemaType.KEYWORD),
            ("main_category", qmodels.PayloadSchemaType.KEYWORD),
            ("source_category", qmodels.PayloadSchemaType.KEYWORD),
            ("source_file", qmodels.PayloadSchemaType.KEYWORD),
            ("price", qmodels.PayloadSchemaType.FLOAT),
        ]
        for field_name, schema in index_plan:
            try:
                self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
                print(f"✅ Payload index created: {field_name}")
            except Exception as e:
                print(f"ℹ️ Skipping/failed index for {field_name}: {e}")

    # =================================================================
    # EMBEDDINGS
    # =================================================================

    def _text_embedding(self, parsed: ParsedQuery) -> List[float]:
        prompts = self._build_clip_prompts(parsed)

        with torch.no_grad():
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)

            # First prompt is closest to user intent, so slightly higher weight.
            weights = torch.tensor([1.30] + [1.0] * (features.shape[0] - 1), device=self.device)
            weighted = (features * weights[:, None]).sum(dim=0)
            weighted = weighted / weighted.norm(p=2)

        return weighted.detach().cpu().numpy().tolist()

    def _image_embedding(self, image_file) -> List[float]:
        image = Image.open(image_file).convert("RGB")
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features[0].detach().cpu().numpy().tolist()

    def _build_clip_prompts(self, parsed: ParsedQuery) -> List[str]:
        q = parsed.clean
        category_phrase = self._category_to_phrase(parsed.category_intent)
        category_hint = f" in {category_phrase}" if category_phrase else ""

        prompts = [
            f"a clear product photo of {q}{category_hint}",
            f"an e-commerce catalog image of {q}{category_hint}",
            f"an Amazon product listing image showing {q}",
            f"a studio product image of {q}",
        ]

        expanded_phrase = " ".join(parsed.expanded_tokens[:16])
        if expanded_phrase and expanded_phrase != q:
            prompts.append(f"a product photo of {expanded_phrase}{category_hint}")

        if parsed.colors:
            prompts.append(f"a {parsed.colors[0]} colored product photo of {q}{category_hint}")

        return prompts

    # =================================================================
    # QUERY UNDERSTANDING
    # =================================================================

    def _parse_query(self, query: str) -> ParsedQuery:
        raw = query or ""
        clean = self._normalize_text(raw)
        tokens = self._tokenize(clean)
        expanded_tokens = self._expand_tokens(tokens)
        min_price, max_price = self._extract_price_range(clean)
        category_intent = self._infer_category(clean, expanded_tokens)
        colors = [t for t in tokens if t in COLORS]
        gender_terms = [t for t in tokens if t in GENDER_TERMS]

        must_terms = [
            t for t in tokens
            if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()
        ]

        return ParsedQuery(
            raw=raw,
            clean=clean,
            tokens=tokens,
            expanded_tokens=expanded_tokens,
            category_intent=category_intent,
            min_price=min_price,
            max_price=max_price,
            colors=colors,
            gender_terms=gender_terms,
            must_terms=must_terms,
        )

    def _normalize_text(self, text: Any) -> str:
        if text is None:
            return ""
        if isinstance(text, (list, tuple, set)):
            text = " ".join(map(str, text))
        elif isinstance(text, dict):
            text = " ".join(f"{k} {v}" for k, v in text.items())
        else:
            text = str(text)

        text = text.lower().strip()
        text = text.replace("₹", " rs ").replace("$", " dollar ")
        text = text.replace("&", " and ")
        text = re.sub(r"[^a-z0-9\s\.\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        raw_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text)
        tokens: List[str] = []
        for tok in raw_tokens:
            if tok in STOPWORDS or len(tok) <= 1:
                continue
            tokens.append(self._simple_stem(tok))
        return tokens

    def _simple_stem(self, token: str) -> str:
        token = token.lower().strip()
        irregular = {
            "mens": "men",
            "womens": "women",
            "ladies": "women",
            "children": "child",
            "knives": "knife",
        }
        if token in irregular:
            return irregular[token]
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("es") and len(token) > 4 and not token.endswith("ses"):
            return token[:-2]
        if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            return token[:-1]
        return token

    def _expand_tokens(self, tokens: List[str]) -> List[str]:
        expanded: List[str] = []
        seen = set()

        for tok in tokens:
            candidates = [tok] + SYNONYMS.get(tok, [])
            for candidate in candidates:
                candidate_norm = self._simple_stem(self._normalize_text(candidate))
                if candidate_norm and candidate_norm not in seen:
                    expanded.append(candidate_norm)
                    seen.add(candidate_norm)
        return expanded

    def _extract_price_range(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        min_price, max_price = None, None

        # between 500 and 1000 / from 500 to 1000
        between = re.search(
            r"(?:between|from)\s+(?:rs\s*)?(\d+(?:\.\d+)?)\s+(?:and|to|-)\s+(?:rs\s*)?(\d+(?:\.\d+)?)",
            text,
        )
        if between:
            a, b = float(between.group(1)), float(between.group(2))
            return min(a, b), max(a, b)

        # 500-1000
        hyphen = re.search(r"\b(?:rs\s*)?(\d+(?:\.\d+)?)\s*-\s*(?:rs\s*)?(\d+(?:\.\d+)?)\b", text)
        if hyphen:
            a, b = float(hyphen.group(1)), float(hyphen.group(2))
            # Avoid treating model numbers like 13-15 as price if very small.
            if max(a, b) > 50:
                return min(a, b), max(a, b)

        upper = re.search(
            r"(?:under|below|less than|upto|up to|max|maximum)\s+(?:rs\s*)?(\d+(?:\.\d+)?)",
            text,
        )
        if upper:
            max_price = float(upper.group(1))

        lower = re.search(
            r"(?:above|over|more than|min|minimum)\s+(?:rs\s*)?(\d+(?:\.\d+)?)",
            text,
        )
        if lower:
            min_price = float(lower.group(1))

        return min_price, max_price

    def _infer_category(self, clean_query: str, expanded_tokens: List[str]) -> Optional[str]:
        token_set = set(expanded_tokens)
        scores: Dict[str, int] = {}

        for category, aliases in CATEGORY_ALIASES.items():
            score = 0
            for alias in aliases:
                alias_clean = self._normalize_text(alias)
                alias_tokens = set(self._tokenize(alias_clean))
                if alias_clean and alias_clean in clean_query:
                    score += 3
                elif alias_tokens and alias_tokens.intersection(token_set):
                    score += 1
            if score > 0:
                scores[category] = score

        if not scores:
            return None
        return max(scores.items(), key=lambda x: x[1])[0]

    # =================================================================
    # QDRANT RETRIEVAL + RERANKING
    # =================================================================

    def _execute_qdrant_query(
        self,
        vector: List[float],
        top_k: int,
        fetch_k: Optional[int] = None,
        parsed_query: Optional[ParsedQuery] = None,
    ) -> List[Dict[str, Any]]:
        top_k = max(1, min(int(top_k), 50))
        fetch_limit = fetch_k or self._dynamic_fetch_k(top_k, parsed_query)

        query_filter = self._build_qdrant_filter(parsed_query) if self.enable_payload_filter else None
        response = self._safe_qdrant_query(vector=vector, limit=fetch_limit, query_filter=query_filter)

        candidates: List[Dict[str, Any]] = []
        for rank, hit in enumerate(response.points, start=1):
            payload = hit.payload or {}
            candidates.append(self._format_hit(hit=hit, rank=rank, payload=payload, parsed_query=parsed_query))

        if not candidates:
            return []

        # Normalize semantic score inside current candidate pool.
        self._add_normalized_semantic_scores(candidates)

        for item in candidates:
            item["hybrid_sort_score"] = self._final_hybrid_score(item, parsed_query)

        candidates.sort(key=lambda x: x["hybrid_sort_score"], reverse=True)
        diversified = self._diversify(candidates, top_k=top_k)
        return [self._public_result(item) for item in diversified]

    def _safe_qdrant_query(
        self,
        vector: List[float],
        limit: int,
        query_filter: Optional[qmodels.Filter] = None,
    ):
        try:
            kwargs = {
                "collection_name": self.collection_name,
                "query": vector,
                "limit": limit,
                "with_payload": True,
            }
            if query_filter is not None:
                kwargs["query_filter"] = query_filter
            return self.qdrant.query_points(**kwargs)
        except Exception as e:
            if query_filter is not None:
                print(f"⚠️ Filtered Qdrant query failed, retrying without filter: {e}")
                return self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=limit,
                    with_payload=True,
                )
            raise

    def _dynamic_fetch_k(self, top_k: int, parsed_query: Optional[ParsedQuery]) -> int:
        # Low-data setting: fetch deep enough for reranker, but not always huge.
        if parsed_query is None:
            return max(80, top_k * 20)

        complexity = len(parsed_query.must_terms)
        if parsed_query.category_intent:
            complexity += 2
        if parsed_query.min_price is not None or parsed_query.max_price is not None:
            complexity += 2
        if parsed_query.colors:
            complexity += 1

        if complexity >= 7:
            return max(350, top_k * 60)
        if complexity >= 4:
            return max(250, top_k * 45)
        return max(160, top_k * 30)

    def _build_qdrant_filter(self, parsed: Optional[ParsedQuery]) -> Optional[qmodels.Filter]:
        """
        Strict filter is optional. Keep enable_payload_filter=False first.
        If your Qdrant payload has clean category/price, you can turn it on.
        """
        if parsed is None:
            return None

        must_conditions: List[Any] = []

        if parsed.category_intent:
            variants = CATEGORY_VALUE_VARIANTS.get(parsed.category_intent, [parsed.category_intent])
            category_fields = ["category", "main_category", "source_category", "source_file"]
            should_conditions = []
            for field_name in category_fields:
                try:
                    should_conditions.append(
                        qmodels.FieldCondition(
                            key=field_name,
                            match=qmodels.MatchAny(any=variants),
                        )
                    )
                except Exception:
                    # Older qdrant-client fallback will be handled by query fallback.
                    pass
            if should_conditions:
                must_conditions.append(qmodels.Filter(should=should_conditions))

        if parsed.min_price is not None or parsed.max_price is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="price",
                    range=qmodels.Range(gte=parsed.min_price, lte=parsed.max_price),
                )
            )

        if not must_conditions:
            return None
        return qmodels.Filter(must=must_conditions)

    def _format_hit(
        self,
        hit: Any,
        rank: int,
        payload: Dict[str, Any],
        parsed_query: Optional[ParsedQuery],
    ) -> Dict[str, Any]:
        title = self._payload_first(payload, ["title", "product_title", "name"])
        brand = self._payload_first(payload, ["brand", "store", "manufacturer"])
        category = self._payload_category(payload)
        description = self._payload_text(payload, ["description", "about_product", "features", "feature", "bullet_points", "details"])
        price = self._payload_first(payload, ["price", "actual_price", "discounted_price", "final_price"])
        image_url = self._payload_image_url(payload)

        item = {
            "rank": rank,
            "raw_semantic_score": float(hit.score),
            "semantic_score": 0.0,  # filled later after candidate normalization
            "title": str(title or ""),
            "brand": str(brand or ""),
            "category": str(category or ""),
            "description": str(description or ""),
            "price": price,
            "price_float": self._parse_price_value(price),
            "image_url": image_url,
            "product_id": self._payload_first(payload, ["product_id", "parent_asin", "asin", "id"]),
            "rating": self._payload_first(payload, ["rating", "average_rating", "stars"]),
            "reviews": self._payload_first(payload, ["reviews", "rating_count", "rating_number", "review_count", "ratings_total"]),
            "raw_payload_category": category,
        }

        item["lexical_score"] = self._lexical_score(parsed_query, item)
        item["category_score"] = self._category_score(parsed_query, item)
        item["price_score"] = self._price_score(parsed_query, item["price"])
        item["attribute_score"] = self._attribute_score(parsed_query, item)
        item["quality_score"] = self._quality_score(item)
        item["rank_score"] = 1.0 / math.sqrt(rank + 1)
        return item

    def _add_normalized_semantic_scores(self, candidates: List[Dict[str, Any]]) -> None:
        scores = [x["raw_semantic_score"] for x in candidates]
        lo, hi = min(scores), max(scores)
        gap = hi - lo
        for item in candidates:
            if gap < 1e-9:
                item["semantic_score"] = 1.0
            else:
                item["semantic_score"] = (item["raw_semantic_score"] - lo) / gap

    def _final_hybrid_score(self, item: Dict[str, Any], parsed: Optional[ParsedQuery]) -> float:
        if parsed is None:
            score = (
                0.82 * item["semantic_score"]
                + 0.10 * item["quality_score"]
                + 0.08 * item["rank_score"]
            )
            return round(float(score), 6)

        score = (
            0.50 * item["semantic_score"]
            + 0.23 * item["lexical_score"]
            + 0.11 * item["category_score"]
            + 0.07 * item["attribute_score"]
            + 0.05 * item["price_score"]
            + 0.03 * item["quality_score"]
            + 0.01 * item["rank_score"]
        )

        # Soft demotion, not removal, for low-data safety.
        if not self._price_in_range(parsed, item.get("price")):
            score -= 0.10

        # If category is clearly wrong, demote but don't delete.
        if parsed.category_intent and item["category_score"] <= 0.05:
            score -= 0.08

        return round(float(score), 6)

    # =================================================================
    # RERANKING FEATURES
    # =================================================================

    def _lexical_score(self, parsed: Optional[ParsedQuery], item: Dict[str, Any]) -> float:
        if parsed is None or not parsed.expanded_tokens:
            return 0.0

        title = self._normalize_text(item.get("title", ""))
        brand = self._normalize_text(item.get("brand", ""))
        category = self._normalize_text(item.get("category", ""))
        description = self._normalize_text(item.get("description", ""))
        all_text = f"{title} {brand} {category} {description}"

        score = 0.0
        max_score = 0.0

        for tok in parsed.expanded_tokens:
            if len(tok) < 2:
                continue
            local = 0.0
            local += 1.00 if self._contains_word_or_phrase(title, tok) else 0.0
            local += 0.80 if self._contains_word_or_phrase(brand, tok) else 0.0
            local += 0.65 if self._contains_word_or_phrase(category, tok) else 0.0
            local += 0.35 if self._contains_word_or_phrase(description, tok) else 0.0
            score += min(local, 1.45)
            max_score += 1.45

        if parsed.clean and parsed.clean in title:
            score += 2.0
            max_score += 2.0

        if parsed.must_terms:
            covered = sum(1 for t in parsed.must_terms if self._contains_word_or_phrase(all_text, t))
            coverage = covered / max(1, len(parsed.must_terms))
            score += coverage
            max_score += 1.0

        return self._clip01(score / max(1e-9, max_score))

    def _category_score(self, parsed: Optional[ParsedQuery], item: Dict[str, Any]) -> float:
        if parsed is None or not parsed.category_intent:
            return 0.0

        combined = self._normalize_text(
            f"{item.get('category', '')} {item.get('title', '')} {item.get('description', '')}"
        )
        category = parsed.category_intent
        variants = [self._normalize_text(v) for v in CATEGORY_VALUE_VARIANTS.get(category, [])]
        aliases = [self._normalize_text(v) for v in CATEGORY_ALIASES.get(category, [])]

        hits = 0
        for v in variants:
            if v and v in combined:
                hits += 2
                break

        for alias in aliases:
            if alias and alias in combined:
                hits += 1
                if hits >= 4:
                    break

        return self._clip01(hits / 4.0)

    def _attribute_score(self, parsed: Optional[ParsedQuery], item: Dict[str, Any]) -> float:
        if parsed is None:
            return 0.0

        text = self._normalize_text(
            f"{item.get('title', '')} {item.get('brand', '')} {item.get('category', '')} {item.get('description', '')}"
        )

        total = 0
        matched = 0

        for color in parsed.colors:
            total += 1
            if self._contains_word_or_phrase(text, color):
                matched += 1

        for gender in parsed.gender_terms:
            total += 1
            gender_aliases = {
                "men": ["men", "mens", "male", "man"],
                "women": ["women", "womens", "female", "woman", "ladies"],
                "boys": ["boy", "boys"],
                "girls": ["girl", "girls"],
                "unisex": ["unisex"],
            }.get(gender, [gender])
            if any(self._contains_word_or_phrase(text, g) for g in gender_aliases):
                matched += 1

        if total == 0:
            return 0.0
        return matched / total

    def _price_score(self, parsed: Optional[ParsedQuery], price_value: Any) -> float:
        if parsed is None:
            return 0.0
        if parsed.min_price is None and parsed.max_price is None:
            return 0.0

        price = self._parse_price_value(price_value)
        if price is None:
            return 0.20

        if parsed.min_price is not None and price < parsed.min_price:
            return 0.0
        if parsed.max_price is not None and price > parsed.max_price:
            distance = price - parsed.max_price
            return max(0.0, 1.0 - distance / max(1.0, parsed.max_price))

        if parsed.max_price is not None:
            # Inside budget: cheaper gets a small boost, but not too much.
            return 0.70 + 0.30 * (1.0 - min(price / max(parsed.max_price, 1.0), 1.0))
        return 1.0

    def _quality_score(self, item: Dict[str, Any]) -> float:
        rating = self._safe_float(item.get("rating"))
        reviews = self._safe_float(item.get("reviews"))

        rating_score = 0.0 if rating is None else self._clip01(rating / 5.0)
        review_score = 0.0
        if reviews is not None and reviews > 0:
            review_score = self._clip01(math.log1p(reviews) / math.log1p(10000))

        if rating is None and reviews is None:
            return 0.0
        return 0.70 * rating_score + 0.30 * review_score

    # =================================================================
    # DIVERSITY / DEDUPLICATION
    # =================================================================

    def _diversify(self, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        seen_keys = set()

        for item in candidates:
            key = self._dedupe_key(item)
            if key in seen_keys:
                continue

            too_similar = False
            for chosen in selected:
                if self._title_jaccard(item.get("title", ""), chosen.get("title", "")) >= 0.82:
                    too_similar = True
                    break
            if too_similar:
                continue

            selected.append(item)
            seen_keys.add(key)
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            selected_ids = {id(x) for x in selected}
            for item in candidates:
                if id(item) not in selected_ids:
                    selected.append(item)
                if len(selected) >= top_k:
                    break

        return selected[:top_k]

    def _dedupe_key(self, item: Dict[str, Any]) -> str:
        product_id = item.get("product_id")
        if product_id:
            return f"id:{product_id}"

        title = self._normalize_text(item.get("title", ""))
        title = re.sub(r"\b\d+(?:gb|ml|kg|g|cm|mm|inch|in|pack|pcs|piece)\b", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        return "title:" + hashlib.md5(title[:140].encode("utf-8")).hexdigest()

    def _title_jaccard(self, a: str, b: str) -> float:
        a_tokens = set(self._tokenize(self._normalize_text(a)))
        b_tokens = set(self._tokenize(self._normalize_text(b)))
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    # =================================================================
    # PAYLOAD EXTRACTION HELPERS
    # =================================================================

    def _payload_first(self, payload: Dict[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            value = payload.get(key)
            if value not in (None, "", [], {}):
                return value
        return None

    def _payload_text(self, payload: Dict[str, Any], keys: Sequence[str]) -> str:
        chunks: List[str] = []
        for key in keys:
            value = payload.get(key)
            if value in (None, "", [], {}):
                continue
            chunks.append(self._flatten_to_text(value))
        return " ".join(chunks)

    def _payload_category(self, payload: Dict[str, Any]) -> Any:
        return self._payload_first(
            payload,
            ["category", "main_category", "source_category", "source_file", "category_name", "categories"],
        )

    def _payload_image_url(self, payload: Dict[str, Any]) -> Optional[str]:
        direct = self._payload_first(payload, ["image_url", "image", "img_url", "thumbnail"])
        if isinstance(direct, str) and direct.startswith("http"):
            return direct

        images = payload.get("images")
        return self._first_url_from_any(images)

    def _first_url_from_any(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value if value.startswith("http") else None
        if isinstance(value, dict):
            # Amazon metadata often stores images as {'large': [...], 'hi_res': [...]}.
            preferred_keys = ["large", "hi_res", "hiRes", "thumb", "variant", "main"]
            for key in preferred_keys:
                url = self._first_url_from_any(value.get(key))
                if url:
                    return url
            for v in value.values():
                url = self._first_url_from_any(v)
                if url:
                    return url
        if isinstance(value, list):
            for x in value:
                url = self._first_url_from_any(x)
                if url:
                    return url
        return None

    def _flatten_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (list, tuple, set)):
            return " ".join(self._flatten_to_text(v) for v in value)
        if isinstance(value, dict):
            return " ".join(f"{k} {self._flatten_to_text(v)}" for k, v in value.items())
        return str(value)

    # =================================================================
    # OUTPUT / UTILITIES
    # =================================================================

    def _public_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "hybrid_sort_score": item["hybrid_sort_score"],
            "confidence_score": round(float(item["raw_semantic_score"]), 3),
            "semantic_rank_score": round(float(item["semantic_score"]), 3),
            "lexical_score": round(float(item["lexical_score"]), 3),
            "category_score": round(float(item["category_score"]), 3),
            "price_score": round(float(item["price_score"]), 3),
            "attribute_score": round(float(item["attribute_score"]), 3),
            "title": item["title"],
            "brand": item.get("brand"),
            "price": item.get("price"),
            "category": item.get("category"),
            "rating": item.get("rating"),
            "reviews": item.get("reviews"),
            "image_url": item.get("image_url"),
        }

    def _contains_word_or_phrase(self, text: str, token_or_phrase: str) -> bool:
        if not text or not token_or_phrase:
            return False
        token_or_phrase = self._normalize_text(token_or_phrase)
        if " " in token_or_phrase:
            return token_or_phrase in text
        token = re.escape(token_or_phrase)
        return re.search(rf"(^|\s){token}(\s|$)", text) is not None

    def _category_to_phrase(self, category: Optional[str]) -> str:
        if not category:
            return ""
        return category.replace("_", " ").lower()

    def _parse_price_value(self, price_value: Any) -> Optional[float]:
        return self._safe_float(price_value)

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            value = str(value).replace(",", "").strip()
            value = re.sub(r"[^0-9\.\-]", "", value)
            if value in ("", ".", "-"):
                return None
            return float(value)
        except Exception:
            return None

    def _price_in_range(self, parsed: ParsedQuery, price_value: Any) -> bool:
        if parsed.min_price is None and parsed.max_price is None:
            return True
        price = self._parse_price_value(price_value)
        if price is None:
            return True
        if parsed.min_price is not None and price < parsed.min_price:
            return False
        if parsed.max_price is not None and price > parsed.max_price:
            return False
        return True

    def _clip01(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    # =================================================================
    # CACHE
    # =================================================================

    def _connect_redis_safely(self):
        host = os.getenv("REDIS_HOST")
        password = os.getenv("REDIS_PASSWORD")
        port = int(os.getenv("REDIS_PORT", "6379"))

        if not host:
            print("ℹ️ REDIS_HOST not found. Running without cache.")
            return None

        try:
            client = redis.Redis(
                host=host,
                port=port,
                password=password,
                decode_responses=True,
                ssl=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            client.ping()
            print("✅ Redis connected.")
            return client
        except Exception as e:
            print(f"⚠️ Redis unavailable. Running without cache: {e}")
            return None

    def _cache_key(self, mode: str, query: str, top_k: int, fetch_k: Optional[int]) -> str:
        normalized = self._normalize_text(query)
        body = json.dumps(
            {
                "v": self.CACHE_VERSION,
                "mode": mode,
                "query": normalized,
                "top_k": top_k,
                "fetch_k": fetch_k,
                "collection": self.collection_name,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:32]
        return f"search:{self.CACHE_VERSION}:{digest}"

    def _read_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        if self.redis_client is None:
            return None
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"⚠️ Redis read error: {e}")
        return None

    def _write_cache(self, cache_key: str, payload: List[Dict[str, Any]]) -> None:
        if self.redis_client is None:
            return
        try:
            self.redis_client.setex(cache_key, self.redis_ttl_seconds, json.dumps(payload))
        except Exception as e:
            print(f"⚠️ Redis write error: {e}")


# =====================================================================
# QUICK USAGE
# =====================================================================
# service = SearchService(enable_payload_filter=False)
# print(service.search("black womens shoes under 2000", top_k=6))
# print(service.search("sunscreen for oily skin", top_k=6))
# print(service.search("yoga mat", top_k=6))
# print(service.search("kitchen storage container", top_k=6))
