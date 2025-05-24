import copy
import json
from dataclasses import dataclass
from typing import Iterable
import torch
from openai.types.chat import ChatCompletionMessageParam
from typing_extensions import final

from src.libs.tokenizer import count_tokens
from src.ffi.bert_token_counter import main as token_counter
from .config import llm, retriever, transformer


@dataclass
class StructuredQuery:
    """Structured representation of enhanced query components."""

    function_names: list[str]
    class_names: list[str]
    variable_names: list[str]
    api_patterns: list[str]
    concepts: list[str]
    original_query: str


@dataclass
class SearchResult:
    """Enhanced search result with scoring."""

    documents: list[str]
    metadatas: list[dict[str, str | int | float | bool | None]]
    distances: list[float]
    source: str  # 'original', 'structured', 'hyde', 'multi_query'


@final
class CGRAG:
    """
    Enhanced CG-RAG with structured queries, HyDE, multi-query support, and strict context grounding.
    """

    def __init__(
        self,
        use_cgrag: bool = True,
        print_cgrag: bool = False,
        context_size: int = 8192,
        context_file_ratio: float = 0.7,
        use_hyde: bool = False,
        use_multi_query: bool = True,
        use_structured_query: bool = True,
    ) -> None:
        self.system_prompt = self.create_system_prompt()

        self.chat_history: list[ChatCompletionMessageParam] = [
            {
                'role': 'system',
                'content': self.system_prompt,
            }
        ]
        self.use_cgrag = use_cgrag
        self.print_cgrag = print_cgrag
        self.use_hyde = use_hyde
        self.use_multi_query = use_multi_query
        self.use_structured_query = use_structured_query
        self.retriever = retriever
        self.transformer = transformer
        self.llm = llm
        self.context_size = context_size
        self.context_file_ratio = context_file_ratio

    def create_system_prompt(self) -> str:
        """Creates a system prompt that enforces strict grounding in retrieved context."""
        return """You are a code analysis assistant that MUST operate strictly within the bounds of provided context.

CRITICAL CONSTRAINTS:
1. You can ONLY discuss code, functions, classes, and variables that are explicitly present in the provided context
2. If asked about something NOT in the context, you MUST say "I don't see that in the provided code context"
3. You MUST quote relevant code snippets when discussing functionality
4. You MUST NOT invent, assume, or hallucinate any code elements

RESPONSE STRUCTURE:
1. First, identify what specific code elements from the context relate to the query
2. Quote the relevant code sections with file paths and line numbers when available
3. Provide analysis ONLY based on what you can see in the quoted code
4. If the context is insufficient, explicitly state what information is missing

FORBIDDEN BEHAVIORS:
- Suggesting code that isn't in the context
- Assuming functionality that isn't visible
- Discussing "typical patterns" unless they're evident in the provided code
- Making recommendations about code you can't see
- Using phrases like "typically", "usually", "commonly", "you should", "consider adding"

When the context is empty or insufficient, respond: "I don't have enough code context to answer this question accurately. Please ensure the relevant code files are indexed."

Remember: You are a code search and analysis tool, not a code generator or general programming advisor."""

    def create_structured_query_prompt(self, base_prompt: str) -> str:
        """Creates a prompt for structured query extraction."""
        return f"""Analyze the user query and extract code-specific information in JSON format.

User Query: "{base_prompt}"

Extract and categorize relevant code elements. Respond with ONLY a valid JSON object in this exact format:

{{
    "function_names": ["exact function names, method names, or signatures"],
    "class_names": ["class names, interface names, type names"],
    "variable_names": ["variable names, parameter names, field names"],
    "api_patterns": ["library.method calls", "framework patterns", "import statements"],
    "concepts": ["algorithmic patterns", "design patterns", "logical operations"]
}}

Examples:
- For "find user authentication logic": 
  {{"function_names": ["authenticate_user", "login", "verify_password"], "class_names": ["User", "AuthService"], "variable_names": ["username", "password", "token"], "api_patterns": ["bcrypt.check", "jwt.encode"], "concepts": ["authentication flow", "password validation"]}}

- For "database connection pooling":
  {{"function_names": ["create_pool", "get_connection"], "class_names": ["ConnectionPool", "Database"], "variable_names": ["pool_size", "max_connections"], "api_patterns": ["sqlalchemy.create_engine", "psycopg2.pool"], "concepts": ["connection pooling", "resource management"]}}

Focus on terms that would appear in actual code. Be specific and precise."""

    def create_hyde_prompt(self, base_prompt: str) -> str:
        """Creates a prompt for hypothetical document generation."""
        return f"""Generate a hypothetical code snippet that would perfectly answer this query: "{base_prompt}"

Create realistic, well-structured code that demonstrates the concept being asked about. Include:
- Proper function/class definitions
- Meaningful variable names
- Relevant comments
- Typical patterns and conventions
- Import statements if relevant

The code should be complete enough to be functional but focused on the core concept. 
This will be used for semantic similarity matching, so make it representative of what the user is looking for.

Example format:
```python
# Relevant imports
import library

class ExampleClass:
    def relevant_method(self, param: type) -> return_type:
        # Implementation that demonstrates the concept
        pass
```"""

    def create_multi_query_prompt(self, base_prompt: str) -> str:
        """Creates a prompt for generating multiple query variations."""
        return f"""Generate 4 different variations of this query optimized for code search: "{base_prompt}"

Each variation should approach the search from a different angle:
1. Technical/Implementation focused
2. Conceptual/Pattern focused  
3. Specific function/method focused
4. Broader context/Architecture focused

Format as a simple numbered list:
1. [First variation]
2. [Second variation] 
3. [Third variation]
4. [Fourth variation]

Make each variation concise but distinct, targeting different aspects of the codebase that might contain relevant information."""

    def validate_context_relevance(self, query: str, context: str) -> bool:
        """Quick check if retrieved context is actually relevant."""
        if not context.strip():
            return False

        # Simple keyword overlap check
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        overlap = len(query_words.intersection(context_words))

        return overlap > 0 or len(context) > 100  # Adjust threshold as needed

    def validate_response_grounding(self, response: str, context: str) -> str:
        """Validate that response doesn't hallucinate beyond context."""
        # Check for common hallucination patterns
        hallucination_indicators = [
            'typically',
            'usually',
            'commonly',
            'often',
            'you should',
            'consider adding',
            'you might want',
            "here's how you could",
            'try implementing',
            'generally',
            'normally',
            'standard practice',
            'best practice',
            'recommended approach',
        ]

        response_lower = response.lower()
        context_lower = context.lower() if context else ''

        flagged_phrases = []
        for indicator in hallucination_indicators:
            if indicator in response_lower and indicator not in context_lower:
                flagged_phrases.append(indicator)

        if flagged_phrases and self.print_cgrag:
            print(
                f'Potential hallucination indicators detected: {flagged_phrases}'
            )

        # Add warning if multiple indicators found
        if len(flagged_phrases) >= 2:
            return f'[NOTICE: Response may contain suggestions beyond the provided code context]\n\n{response}'

        return response

    def parse_structured_query(self, llm_response: str) -> StructuredQuery:
        """Parse LLM response into structured query components."""
        try:
            # Extract JSON from response
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1

            if start_idx == -1 or end_idx <= start_idx:
                if self.print_cgrag:
                    print('No valid JSON found in structured query response')
                return StructuredQuery([], [], [], [], [], '')

            json_str = llm_response[start_idx:end_idx]
            parsed: dict[str, list[str]] = json.loads(json_str)

            return StructuredQuery(
                function_names=parsed.get('function_names', []),
                class_names=parsed.get('class_names', []),
                variable_names=parsed.get('variable_names', []),
                api_patterns=parsed.get('api_patterns', []),
                concepts=parsed.get('concepts', []),
                original_query='',
            )
        except json.JSONDecodeError as e:
            if self.print_cgrag:
                print(f'JSON parsing error: {e}')
            return StructuredQuery([], [], [], [], [], '')
        except Exception as e:
            if self.print_cgrag:
                print(f'Error parsing structured query: {e}')
            return StructuredQuery([], [], [], [], [], '')

    def parse_multi_queries(self, llm_response: str) -> list[str]:
        """Parse LLM response into multiple query variations."""
        try:
            queries = []
            lines = llm_response.strip().split('\n')

            for line in lines:
                line = line.strip()
                # Look for numbered list format: "1. query", "2. query", etc.
                if line and (line[0].isdigit() or line.startswith('- ')):
                    # Remove the number/bullet and extract the query
                    if '. ' in line:
                        query = line.split('. ', 1)[1].strip()
                    elif line.startswith('- '):
                        query = line[2:].strip()
                    else:
                        query = line

                    if query:
                        queries.append(query)

            return queries[:4]  # Limit to 4 variations
        except Exception as e:
            if self.print_cgrag:
                print(f'Error parsing multi-queries: {e}')
            return []

    def structured_query_to_string(
        self, structured_query: StructuredQuery
    ) -> str:
        """Convert structured query back to a search string."""
        all_terms = []

        # Prioritize function and class names (most specific)
        all_terms.extend(structured_query.function_names)
        all_terms.extend(structured_query.class_names)

        # Add other terms
        all_terms.extend(structured_query.variable_names)
        all_terms.extend(structured_query.api_patterns)
        all_terms.extend(structured_query.concepts)

        # Create a comprehensive search string
        if all_terms:
            return ' '.join(all_terms)
        return ''

    def retrieve_with_query(
        self, query: str, source: str = 'original', n_results: int = 50
    ) -> SearchResult:
        """Retrieve documents for a single query."""
        try:
            encoded_query: torch.Tensor = self.transformer.encode(query)
            results = self.retriever.query(
                query_embeddings=encoded_query.tolist(),
                n_results=n_results,
            )

            if not results:
                return SearchResult([], [], [], source)

            documents = results.get('documents', [[]])
            metadatas = results.get('metadatas', [[]])
            distances = results.get('distances', [[]])

            if not documents or not documents[0]:
                return SearchResult([], [], [], source)

            # Convert metadatas to the correct type
            typed_metadatas: list[
                dict[str, str | int | float | bool | None]
            ] = []
            if metadatas and metadatas[0]:
                for metadata in metadatas[0]:
                    if isinstance(metadata, dict):
                        typed_metadatas.append(metadata)
                    else:
                        typed_metadatas.append({})

            return SearchResult(
                documents=documents[0],
                metadatas=typed_metadatas,
                distances=distances[0] if distances and distances[0] else [],
                source=source,
            )

        except Exception as e:
            if self.print_cgrag:
                print(f'Error in retrieve_with_query for {source}: {e}')
            return SearchResult([], [], [], source)

    def aggregate_search_results(
        self, results: list[SearchResult]
    ) -> dict[
        str,
        list[str]
        | list[dict[str, str | int | float | bool | None]]
        | list[float],
    ]:
        """Aggregate and deduplicate results from multiple searches."""
        seen_content = set()
        aggregated_docs = []
        aggregated_metadata = []
        aggregated_scores = []  # Combined scoring

        # Weight different sources
        source_weights = {
            'structured': 1.0,  # Highest priority
            'hyde': 0.9,
            'multi_query': 0.8,
            'original': 0.7,  # Lowest priority
        }

        # Score based on distance (lower is better) and source weight
        scored_items: list[
            tuple[float, str, dict[str, str | int | float | bool | None]]
        ] = []

        for result in results:
            weight = source_weights.get(result.source, 0.5)

            # Explicit type annotations for zip unpacking
            doc: str
            metadata: dict[str, str | int | float | bool | None]
            distance: float

            for doc, metadata, distance in zip(
                result.documents,
                result.metadatas,
                result.distances,
                strict=True,
            ):
                # Create a hash of the document content for deduplication
                content_hash = hash(doc.strip())

                if content_hash not in seen_content:
                    seen_content.add(content_hash)

                    # Lower score is better (distance is inverted, weight is applied)
                    score = distance / weight

                    scored_items.append((score, doc, metadata))

        # Sort by score (lower is better) and take the best results
        scored_items.sort(key=lambda x: x[0])

        # Extract the best results
        max_results = 80  # Adjust based on needs
        for score, doc, metadata in scored_items[:max_results]:
            aggregated_docs.append(doc)
            aggregated_metadata.append(metadata)
            aggregated_scores.append(score)

        return {
            'documents': [aggregated_docs],
            'metadatas': [aggregated_metadata],
            'scores': aggregated_scores,
        }

    def run_llm_for_enhancement(
        self, prompt: str, purpose: str = 'enhancement'
    ) -> str:
        """Run LLM for query enhancement with error handling."""
        try:
            # Create a separate temporary conversation for enhancement
            temp_messages: Iterable[ChatCompletionMessageParam] = [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that follows instructions precisely.',
                },
                {'role': 'user', 'content': prompt},
            ]

            response = self.llm.chat.completions.create(
                model='gpt-4o-mini',
                messages=temp_messages,
                max_tokens=800,
                temperature=0.1,
                stream=False,
            )

            content = response.choices[0].message.content or ''

            if self.print_cgrag:
                print(f'LLM {purpose} response: {content[:200]}...')

            return content

        except Exception as e:
            if self.print_cgrag:
                print(f'Error in LLM {purpose}: {e}')
            return ''

    def enhanced_retrieve(self, query: str) -> str:
        """
        Enhanced retrieval using structured queries, HyDE, and multi-query.
        """
        if self.print_cgrag:
            print(f'Enhanced retrieval for: {query}')

        all_results = []

        # 1. Original query search
        original_result = self.retrieve_with_query(query, 'original')
        if original_result.documents:
            all_results.append(original_result)

        if not self.use_cgrag:
            # If CGRAG is disabled, just use original results
            return self.build_context_with_metadata([original_result])

        # 2. Structured query approach
        if self.use_structured_query:
            try:
                structured_prompt = self.create_structured_query_prompt(query)
                structured_response = self.run_llm_for_enhancement(
                    structured_prompt, 'structured_query'
                )

                if structured_response:
                    structured_query = self.parse_structured_query(
                        structured_response
                    )
                    structured_search_string = self.structured_query_to_string(
                        structured_query
                    )

                    if structured_search_string:
                        structured_result = self.retrieve_with_query(
                            structured_search_string, 'structured'
                        )
                        if structured_result.documents:
                            all_results.append(structured_result)

                        if self.print_cgrag:
                            print(
                                f'Structured search: {structured_search_string[:100]}...'
                            )

            except Exception as e:
                if self.print_cgrag:
                    print(f'Error in structured query: {e}')

        # 3. HyDE approach
        if self.use_hyde:
            try:
                hyde_prompt = self.create_hyde_prompt(query)
                hyde_response = self.run_llm_for_enhancement(
                    hyde_prompt, 'hyde'
                )

                if hyde_response:
                    # Extract code from response (look for code blocks)
                    hyde_code = self.extract_code_from_response(hyde_response)
                    search_text = hyde_code if hyde_code else hyde_response

                    hyde_result = self.retrieve_with_query(search_text, 'hyde')
                    if hyde_result.documents:
                        all_results.append(hyde_result)

                    if self.print_cgrag:
                        print(f'HyDE search with: {search_text[:100]}...')

            except Exception as e:
                if self.print_cgrag:
                    print(f'Error in HyDE: {e}')

        # 4. Multi-query approach
        if self.use_multi_query:
            try:
                multi_query_prompt = self.create_multi_query_prompt(query)
                multi_response = self.run_llm_for_enhancement(
                    multi_query_prompt, 'multi_query'
                )

                if multi_response:
                    query_variations = self.parse_multi_queries(multi_response)

                    for i, variation in enumerate(query_variations):
                        if variation.strip():
                            var_result = self.retrieve_with_query(
                                variation, f'multi_query_{i}'
                            )
                            if var_result.documents:
                                all_results.append(var_result)

                    if self.print_cgrag:
                        print(
                            f'Multi-query variations: {len(query_variations)}'
                        )

            except Exception as e:
                if self.print_cgrag:
                    print(f'Error in multi-query: {e}')

        # 5. Aggregate all results
        if all_results:
            aggregated_results = self.aggregate_search_results(all_results)
            return self.build_context_from_aggregated_results_with_metadata(
                aggregated_results
            )

        return ''

    def extract_code_from_response(self, response: str) -> str:
        """Extract code blocks from LLM response."""
        try:
            # Look for code blocks marked with ```
            if '```' in response:
                code_blocks = []
                lines = response.split('\n')
                in_code_block = False
                current_block = []

                for line in lines:
                    if line.strip().startswith('```'):
                        if in_code_block:
                            # End of code block
                            if current_block:
                                code_blocks.append('\n'.join(current_block))
                            current_block = []
                            in_code_block = False
                        else:
                            # Start of code block
                            in_code_block = True
                    elif in_code_block:
                        current_block.append(line)

                return '\n\n'.join(code_blocks) if code_blocks else response

            return response
        except Exception:
            return response

    def build_context_with_metadata(self, results: list[SearchResult]) -> str:
        """Build context text with explicit file and location information."""
        try:
            relevant_full_text = ''
            chunk_total_tokens = 0
            max_tokens = int(self.context_size * self.context_file_ratio)

            context_sources = []

            for result in results:
                for raw_code, metadata in zip(
                    result.documents, result.metadatas
                ):
                    # Extract file information
                    file_path = metadata.get('file_path', 'unknown')
                    start_line = metadata.get('start_line', 'unknown')
                    end_line = metadata.get('end_line', 'unknown')

                    # Create clear source attribution
                    source_header = f'\n--- SOURCE: {file_path} (lines {start_line}-{end_line}) ---\n'
                    display_text = (
                        source_header + raw_code + '\n--- END SOURCE ---\n'
                    )

                    chunk_tokens = self._estimate_tokens(display_text)

                    if chunk_total_tokens + chunk_tokens >= max_tokens:
                        break

                    chunk_total_tokens += chunk_tokens
                    relevant_full_text += display_text
                    context_sources.append(
                        f'{file_path}:{start_line}-{end_line}'
                    )

            # Add summary of sources at the top
            if context_sources:
                source_summary = (
                    f'CODE SOURCES: {", ".join(set(context_sources))}\n\n'
                )
                relevant_full_text = source_summary + relevant_full_text

            return relevant_full_text.strip()

        except Exception as e:
            if self.print_cgrag:
                print(f'Error in build_context_with_metadata: {e}')
            return ''

    def build_context_from_results(self, results: list[SearchResult]) -> str:
        """Build context text from search results."""
        return self.build_context_with_metadata(results)

    def build_context_from_aggregated_results_with_metadata(
        self,
        aggregated_results: dict[
            str,
            list[str]
            | list[dict[str, str | int | float | bool | None]]
            | list[float],
        ],
    ) -> str:
        """Build context text from aggregated results with metadata."""
        try:
            relevant_full_text = ''
            chunk_total_tokens = 0
            max_tokens = int(self.context_size * self.context_file_ratio)

            documents_data = aggregated_results.get('documents', [[]])
            metadatas_data = aggregated_results.get('metadatas', [[]])

            documents = (
                documents_data[0] if isinstance(documents_data[0], list) else []
            )
            metadatas = (
                metadatas_data[0] if isinstance(metadatas_data[0], list) else []
            )

            context_sources = []

            for raw_code, metadata in zip(documents, metadatas):
                if isinstance(metadata, dict):
                    # Extract file information
                    file_path = metadata.get('file_path', 'unknown')
                    start_line = metadata.get('start_line', 'unknown')
                    end_line = metadata.get('end_line', 'unknown')

                    # Create clear source attribution
                    source_header = f'\n--- SOURCE: {file_path} (lines {start_line}-{end_line}) ---\n'
                    display_text = (
                        source_header + str(raw_code) + '\n--- END SOURCE ---\n'
                    )
                    context_sources.append(
                        f'{file_path}:{start_line}-{end_line}'
                    )
                else:
                    display_text = str(raw_code)

                chunk_tokens = self._estimate_tokens(display_text)

                if chunk_total_tokens + chunk_tokens >= max_tokens:
                    break

                chunk_total_tokens += chunk_tokens
                relevant_full_text += display_text

            # Add summary of sources at the top
            if context_sources:
                source_summary = (
                    f'CODE SOURCES: {", ".join(set(context_sources))}\n\n'
                )
                relevant_full_text = source_summary + relevant_full_text

            return relevant_full_text.strip()

        except Exception as e:
            if self.print_cgrag:
                print(
                    f'Error in build_context_from_aggregated_results_with_metadata: {e}'
                )
            return ''

    def build_context_from_aggregated_results(
        self,
        aggregated_results: dict[
            str,
            list[str]
            | list[dict[str, str | int | float | bool | None]]
            | list[float],
        ],
    ) -> str:
        """Build context text from aggregated results."""
        return self.build_context_from_aggregated_results_with_metadata(
            aggregated_results
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return count_tokens(text)

    def create_user_message(self, content: str) -> ChatCompletionMessageParam:
        """Create a user message for chat history."""
        return {'role': 'user', 'content': content}

    def create_assistant_message(
        self, content: str
    ) -> ChatCompletionMessageParam:
        """Create an assistant message for chat history."""
        return {'role': 'assistant', 'content': content}

    def generate(self, query: str, retrieved_info: str) -> str:
        """Generate a response with strict context grounding."""
        try:
            # Validate context relevance
            if not self.validate_context_relevance(query, retrieved_info):
                if self.print_cgrag:
                    print('Context validation failed - low relevance detected')

            # Create conversation with enforced grounding
            if retrieved_info.strip():
                # Explicit context formatting with clear instructions
                user_content = f"""RETRIEVED CODE CONTEXT:
{retrieved_info}

USER QUERY: {query}

Please analyze the above code context to answer the user's query. Only discuss what you can see in the provided context. If the context doesn't contain enough information to answer the query, explicitly state that."""
            else:
                user_content = f"""No relevant code context was retrieved for this query.

USER QUERY: {query}

Since no code context is available, please inform the user that you cannot provide a code-specific answer and suggest ensuring the relevant code files are indexed."""

            user_message = self.create_user_message(user_content)
            self.chat_history.append(user_message)

            response = self.llm.chat.completions.create(
                model='gpt-4o-mini',
                messages=self.chat_history,
                temperature=0.0,  # Zero temperature for maximum determinism
                max_tokens=1000,
            )

            content = response.choices[0].message.content or ''

            # Validate response for hallucination
            validated_content = self.validate_response_grounding(
                content, retrieved_info
            )

            assistant_message = self.create_assistant_message(validated_content)
            self.chat_history.append(assistant_message)

            return validated_content

        except Exception as e:
            error_msg = f'Error generating response: {e}'
            if self.print_cgrag:
                print(error_msg)
            return error_msg

    def run(self, query: str) -> str:
        """Run the enhanced CGRAG system."""
        if not query.strip():
            return 'Please provide a valid query.'

        try:
            retrieved_info = self.enhanced_retrieve(query)

            if self.print_cgrag:
                print(
                    f'Enhanced retrieval completed. Context length: {len(retrieved_info)}'
                )
                if retrieved_info:
                    print(f'Context preview: {retrieved_info[:200]}...')
                else:
                    print('No context retrieved')

            response = self.generate(query, retrieved_info)
            return response

        except Exception as e:
            error_msg = f'Error in enhanced CGRAG run: {e}'
            if self.print_cgrag:
                print(error_msg)
            return error_msg

    def reset(self) -> None:
        """Reset the CGRAG system to its initial state."""
        self.chat_history = [
            {
                'role': 'system',
                'content': self.system_prompt,
            }
        ]

    def get_enhancement_status(self) -> dict[str, bool]:
        """Get current status of all enhancement features."""
        return {
            'cgrag_enabled': self.use_cgrag,
            'hyde_enabled': self.use_hyde,
            'multi_query_enabled': self.use_multi_query,
            'structured_query_enabled': self.use_structured_query,
            'debug_mode': self.print_cgrag,
        }

    def get_context_quality_stats(self) -> dict[str, int | str]:
        """Get statistics about the last context retrieval."""
        last_message = None
        for msg in reversed(self.chat_history):
            if msg['role'] == 'user':
                last_message = msg
                break

        if not last_message:
            return {'status': 'No recent queries'}

        content = str(last_message.get('content', ''))
        if 'RETRIEVED CODE CONTEXT:' in content:
            context_start = content.find('RETRIEVED CODE CONTEXT:') + len(
                'RETRIEVED CODE CONTEXT:'
            )
            context_end = content.find('USER QUERY:')
            if context_end != -1:
                context = content[context_start:context_end].strip()
                return {
                    'context_length': len(context),
                    'context_tokens': self._estimate_tokens(context),
                    'has_source_attribution': '--- SOURCE:' in context,
                    'source_count': context.count('--- SOURCE:'),
                    'status': 'Context available',
                }

        return {'status': 'No context in last query'}
