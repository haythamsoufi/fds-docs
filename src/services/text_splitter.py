"""Advanced text splitting with adaptive strategies."""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception as _e:  # pragma: no cover
    AutoTokenizer = None  # Fallback if transformers is unavailable


@dataclass
class TextChunk:
    """Text chunk with metadata."""
    content: str
    metadata: Dict[str, Any]
    start_index: int = 0
    end_index: int = 0


class BaseSplitter(ABC):
    """Base class for text splitters."""
    
    @abstractmethod
    async def split(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Split text into chunks."""
        pass


class SemanticSplitter(BaseSplitter):
    """Semantic-aware text splitter."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    async def split(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Split text using semantic boundaries."""
        # Split by paragraphs first
        paragraphs = self._split_by_paragraphs(text)
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Create chunk
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    metadata={
                        **metadata,
                        "chunk_type": "semantic",
                        "paragraph_count": current_chunk.count('\n\n') + 1
                    },
                    start_index=current_start,
                    end_index=current_start + len(current_chunk)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + paragraph
                current_start = current_start + len(current_chunk) - len(overlap_text)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Additional safety check: if single paragraph is too long, split it
            if len(current_chunk) > self.chunk_size * 1.5:  # 1.5x threshold
                # Force create a chunk even if it's larger than ideal
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    metadata={
                        **metadata,
                        "chunk_type": "semantic_forced",
                        "paragraph_count": current_chunk.count('\n\n') + 1
                    },
                    start_index=current_start,
                    end_index=current_start + len(current_chunk)
                )
                chunks.append(chunk)
                current_chunk = ""
                current_start = current_start + len(current_chunk)
                    
        # Add final chunk
        if current_chunk.strip():
            chunk = TextChunk(
                content=current_chunk.strip(),
                metadata={
                    **metadata,
                    "chunk_type": "semantic",
                    "paragraph_count": current_chunk.count('\n\n') + 1
                },
                start_index=current_start,
                end_index=current_start + len(current_chunk)
            )
            chunks.append(chunk)
            
        return chunks
        
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs."""
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
        
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        if len(text) <= self.chunk_overlap:
            return text
            
        # Try to find sentence boundary for overlap
        sentences = re.split(r'[.!?]+', text)
        overlap_text = ""
        
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= self.chunk_overlap:
                overlap_text = sentence + overlap_text
            else:
                break
                
        return overlap_text.strip()


class StructuralSplitter(BaseSplitter):
    """Structure-aware text splitter for documents with clear sections."""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        
    async def split(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Split text by structural elements."""
        # Detect document structure
        sections = self._detect_sections(text)
        chunks = []
        
        for section in sections:
            if len(section['content']) <= self.chunk_size:
                # Section fits in one chunk
                chunk = TextChunk(
                    content=section['content'],
                    metadata={
                        **metadata,
                        "chunk_type": "structural",
                        "section_title": section.get('title', ''),
                        "section_level": section.get('level', 0)
                    },
                    start_index=section['start'],
                    end_index=section['end']
                )
                chunks.append(chunk)
            else:
                # Split large section
                sub_chunks = await self._split_large_section(section, metadata)
                chunks.extend(sub_chunks)
                
        return chunks
        
    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect document sections."""
        sections = []
        
        # Look for common section patterns
        patterns = [
            r'^(#{1,6})\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS headers
            r'^(\d+\.?\s+.+)$',    # Numbered sections
            r'^\[Page \d+\]$'      # Page markers
        ]
        
        lines = text.split('\n')
        current_section = {'content': '', 'start': 0, 'end': 0}
        char_position = 0
        
        for line in lines:
            line_with_newline = line + '\n'
            
            # Check if line matches any header pattern
            is_header = False
            for pattern in patterns:
                if re.match(pattern, line.strip(), re.MULTILINE):
                    # Save previous section
                    if current_section['content'].strip():
                        current_section['end'] = char_position
                        sections.append(current_section.copy())
                        
                    # Start new section
                    current_section = {
                        'content': line_with_newline,
                        'title': line.strip(),
                        'start': char_position,
                        'end': char_position + len(line_with_newline),
                        'level': self._get_header_level(line, pattern)
                    }
                    is_header = True
                    break
                    
            if not is_header:
                current_section['content'] += line_with_newline
                
            char_position += len(line_with_newline)
            
        # Add final section
        if current_section['content'].strip():
            current_section['end'] = char_position
            sections.append(current_section)
            
        return sections
        
    def _get_header_level(self, line: str, pattern: str) -> int:
        """Determine header level."""
        if pattern.startswith(r'^(#{1,6})'):
            return len(re.match(r'^(#+)', line.strip()).group(1))
        elif pattern.startswith(r'^(\d+'):
            return 1
        else:
            return 0
            
    async def _split_large_section(self, section: Dict[str, Any], metadata: Dict[str, Any]) -> List[TextChunk]:
        """Split large section into smaller chunks."""
        # Use semantic splitter for large sections
        semantic_splitter = SemanticSplitter(self.chunk_size)
        sub_chunks = await semantic_splitter.split(section['content'], metadata)
        
        # Update metadata to include section info
        for chunk in sub_chunks:
            chunk.metadata.update({
                "parent_section": section.get('title', ''),
                "section_level": section.get('level', 0)
            })
            
        return sub_chunks


class TokenSentenceSplitter(BaseSplitter):
    """Token-based, sentence-aware splitter with overlap."""
    
    def __init__(self, model_name: str, max_tokens: int, overlap_tokens: int):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.overlap_tokens = max_tokens if overlap_tokens > max_tokens else overlap_tokens
        self._tokenizer = None
        
        if AutoTokenizer is not None:
            try:
                # Use fast tokenizer if available
                self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer for {model_name}: {e}. Falling back to semantic splitter.")
                self._tokenizer = None
        else:
            logger.warning("transformers not available; falling back to semantic splitter.")
            self._tokenizer = None
    
    def _split_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sentences with character spans.
        Returns list of { 'text': str, 'start': int, 'end': int }.
        """
        # Regex to capture sentences including trailing punctuation; supports newlines
        pattern = r"[^.!?]+(?:[.!?](?=\s)|$)"
        sentences: List[Dict[str, Any]] = []
        for match in re.finditer(pattern, text, flags=re.MULTILINE | re.DOTALL):
            sent_text = match.group(0)
            # Skip empty/whitespace-only
            if not sent_text or not sent_text.strip():
                continue
            sentences.append({
                "text": sent_text,
                "start": match.start(),
                "end": match.end()
            })
        return sentences

    def _find_page_spans(self, text: str) -> List[Dict[str, Any]]:
        """Detect page spans based on markers like '[Page N]'."""
        pages: List[Dict[str, Any]] = []
        markers = list(re.finditer(r"\[Page\s+(\d+)\]", text))
        if not markers:
            # Single page fallback spanning entire text
            return [{"number": 1, "start": 0, "end": len(text)}]
        for i, m in enumerate(markers):
            page_num = int(m.group(1))
            start = m.start()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            pages.append({"number": page_num, "start": start, "end": end})
        return pages

    def _detect_sections_for_text(self, text: str) -> List[Dict[str, Any]]:
        """Reuse structural detection to identify section boundaries with titles and levels."""
        try:
            detector = StructuralSplitter()
            sections = detector._detect_sections(text)
            # Ensure required keys exist
            normalized: List[Dict[str, Any]] = []
            for s in sections:
                normalized.append({
                    "title": s.get("title", ""),
                    "level": s.get("level", 0),
                    "start": s.get("start", 0),
                    "end": s.get("end", 0),
                })
            return normalized
        except Exception as _e:
            return []
    
    def _token_count(self, text: str) -> int:
        if not self._tokenizer:
            # Approximate with characters / 4 as a rough heuristic
            return max(1, len(text) // 4)
        return len(self._tokenizer.encode(text, add_special_tokens=False))
    
    async def split(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        # If tokenizer not available, fallback to semantic splitter with smaller chunks
        if self._tokenizer is None:
            # Use much smaller chunks when tokenizer is not available to respect token limits
            smaller_chunk_size = min(300, settings.chunk_size)  # Cap at 300 chars for ~75 tokens
            smaller_overlap = min(50, settings.chunk_overlap)   # Cap at 50 chars for ~12 tokens
            fallback = SemanticSplitter(smaller_chunk_size, smaller_overlap)
            return await fallback.split(text, metadata)

        # Pre-compute structure maps
        page_spans = self._find_page_spans(text)
        sections = self._detect_sections_for_text(text)
        sentences = self._split_sentences(text)
        chunks: List[TextChunk] = []
        current_sentences: List[Dict[str, Any]] = []
        current_tokens = 0
        
        for sent in sentences:
            sentence_text: str = sent["text"]
            sent_tokens = self._token_count(sentence_text)
            if current_sentences and current_tokens + sent_tokens > self.max_tokens:
                # finalize current chunk
                content = "\n\n".join(s["text"] for s in current_sentences).strip()
                chunk_start = current_sentences[0]["start"]
                chunk_end = current_sentences[-1]["end"]
                # Derive page coverage
                covered_pages = [p["number"] for p in page_spans if not (p["end"] <= chunk_start or p["start"] >= chunk_end)]
                page_start = covered_pages[0] if covered_pages else None
                page_end = covered_pages[-1] if covered_pages else None
                # Find primary section at chunk start
                primary_section = None
                for sec in sections:
                    if sec["start"] <= chunk_start < sec.get("end", len(text)):
                        primary_section = sec
                chunk_metadata = {
                    **metadata,
                    "chunk_type": "token_sentence",
                    "sentence_count": len(current_sentences),
                    "token_count": current_tokens,
                    "page_start": page_start,
                    "page_end": page_end,
                    "pages": covered_pages,
                    "section_title": (primary_section.get("title") if primary_section else ""),
                    "section_level": (primary_section.get("level") if primary_section else 0),
                }
                chunk = TextChunk(
                    content=content,
                    metadata=chunk_metadata,
                    start_index=chunk_start,
                    end_index=chunk_end
                )
                chunks.append(chunk)
                
                # prepare overlap for next chunk
                overlap_sentences: List[Dict[str, Any]] = []
                overlap_tokens_accum = 0
                for s in reversed(current_sentences):
                    t = self._token_count(s["text"])
                    if overlap_tokens_accum + t > self.overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens_accum += t
                
                current_sentences = overlap_sentences[:]  # start next with overlap
                current_tokens = overlap_tokens_accum
            
            # add current sentence
            current_sentences.append(sent)
            current_tokens += sent_tokens
        
        if current_sentences:
            content = "\n\n".join(s["text"] for s in current_sentences).strip()
            chunk_start = current_sentences[0]["start"]
            chunk_end = current_sentences[-1]["end"]
            covered_pages = [p["number"] for p in page_spans if not (p["end"] <= chunk_start or p["start"] >= chunk_end)]
            page_start = covered_pages[0] if covered_pages else None
            page_end = covered_pages[-1] if covered_pages else None
            primary_section = None
            for sec in sections:
                if sec["start"] <= chunk_start < sec.get("end", len(text)):
                    primary_section = sec
            chunk = TextChunk(
                content=content,
                metadata={
                    **metadata,
                    "chunk_type": "token_sentence",
                    "sentence_count": len(current_sentences),
                    "token_count": current_tokens,
                    "page_start": page_start,
                    "page_end": page_end,
                    "pages": covered_pages,
                    "section_title": (primary_section.get("title") if primary_section else ""),
                    "section_level": (primary_section.get("level") if primary_section else 0),
                },
                start_index=chunk_start,
                end_index=chunk_end
            )
            chunks.append(chunk)
        
        return chunks


class LegalDocumentSplitter(BaseSplitter):
    """Specialized splitter for legal documents."""
    
    def __init__(self, chunk_size: int = 1500):
        self.chunk_size = chunk_size
        
    async def split(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Split legal document by clauses and sections."""
        # Legal document patterns
        clause_patterns = [
            r'^\s*\d+\.\s+',  # 1. Clause
            r'^\s*\([a-z]\)\s+',  # (a) Sub-clause
            r'^\s*\([ivx]+\)\s+',  # (i) Roman numeral
            r'^\s*Article\s+\d+',  # Article 1
            r'^\s*Section\s+\d+',  # Section 1
        ]
        
        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        current_start = 0
        char_position = 0
        
        for line in lines:
            line_with_newline = line + '\n'
            
            # Check if line starts a new legal clause
            is_new_clause = any(re.match(pattern, line) for pattern in clause_patterns)
            
            if is_new_clause and current_chunk and len(current_chunk) > 100:
                # Create chunk for previous clause
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    metadata={
                        **metadata,
                        "chunk_type": "legal_clause",
                        "clause_markers": self._extract_clause_markers(current_chunk)
                    },
                    start_index=current_start,
                    end_index=char_position
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = line_with_newline
                current_start = char_position
            else:
                current_chunk += line_with_newline
                
            char_position += len(line_with_newline)
            
            # Split if chunk gets too large
            if len(current_chunk) > self.chunk_size:
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    metadata={
                        **metadata,
                        "chunk_type": "legal_clause",
                        "clause_markers": self._extract_clause_markers(current_chunk)
                    },
                    start_index=current_start,
                    end_index=char_position
                )
                chunks.append(chunk)
                current_chunk = ""
                current_start = char_position
                
        # Add final chunk
        if current_chunk.strip():
            chunk = TextChunk(
                content=current_chunk.strip(),
                metadata={
                    **metadata,
                    "chunk_type": "legal_clause",
                    "clause_markers": self._extract_clause_markers(current_chunk)
                },
                start_index=current_start,
                end_index=char_position
            )
            chunks.append(chunk)
            
        return chunks
        
    def _extract_clause_markers(self, text: str) -> List[str]:
        """Extract clause markers from text."""
        patterns = [
            r'\d+\.',
            r'\([a-z]\)',
            r'\([ivx]+\)',
            r'Article\s+\d+',
            r'Section\s+\d+'
        ]
        
        markers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            markers.extend(matches)
            
        return markers


class AdaptiveTextSplitter:
    """Adaptive text splitter that chooses strategy based on content."""
    
    def __init__(self):
        self.semantic_splitter = SemanticSplitter(
            settings.chunk_size, 
            settings.chunk_overlap
        )
        self.structural_splitter = StructuralSplitter(settings.chunk_size)
        self.legal_splitter = LegalDocumentSplitter(settings.chunk_size)
        # Prefer token-based sentence-aware splitter if transformers is available
        # Use configured max_input_tokens or fallback to chunk_size based calculation
        max_tokens = getattr(settings, 'max_input_tokens', None) or max(256, min(512, settings.chunk_size // 3))
        max_context = getattr(settings, 'max_context_tokens', None) or max(200, min(400, max_tokens // 2))
        approx_tokens = min(max_tokens, max_context)  # Use the smaller of the two limits
        approx_overlap = max(32, min(approx_tokens // 5, settings.chunk_overlap // 3 if settings.chunk_overlap else 64))
        self.token_sentence_splitter = TokenSentenceSplitter(
            model_name=settings.embedding_model,
            max_tokens=approx_tokens,
            overlap_tokens=approx_overlap
        )
        
    async def split_text(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Split text using the most appropriate strategy."""
        # Determine document type
        doc_type = self._classify_document_type(text, metadata)
        
        # Prefer token-based splitter generally; fall back by type when appropriate
        if doc_type == "legal":
            return await self.legal_splitter.split(text, metadata)
        if doc_type == "structured":
            # Use token splitter to respect token limits but preserve sections via metadata downstream
            return await self.token_sentence_splitter.split(text, metadata)
        # Default
        return await self.token_sentence_splitter.split(text, metadata)
            
    def _classify_document_type(self, text: str, metadata: Dict[str, Any]) -> str:
        """Classify document type for appropriate splitting strategy."""
        # Check for legal document indicators
        legal_indicators = [
            r'Article\s+\d+',
            r'Section\s+\d+',
            r'Clause\s+\d+',
            r'whereas',
            r'hereby',
            r'aforementioned'
        ]
        
        legal_score = sum(1 for pattern in legal_indicators 
                         if re.search(pattern, text, re.IGNORECASE))
        
        # Check for structured document indicators
        structure_indicators = [
            r'^#{1,6}\s+',  # Markdown headers
            r'^\d+\.\s+',   # Numbered lists
            r'^\[Page \d+\]', # Page markers
            r'^[A-Z][A-Z\s]+$'  # ALL CAPS headers
        ]
        
        structure_score = sum(1 for pattern in structure_indicators 
                            if re.search(pattern, text, re.MULTILINE))
        
        # Decision logic
        if legal_score >= 3:
            return "legal"
        elif structure_score >= 5:
            return "structured"
        else:
            return "semantic"
