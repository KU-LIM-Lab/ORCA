"""Data Preprocessor Agent implementation using SpecialistAgent pattern."""

from typing import Any, Dict, Optional, List
import pandas as pd
import logging

from core.base import SpecialistAgent, AgentType
from core.state import AgentState
from utils.database import Database
from utils.redis_df import load_df_parquet, save_df_parquet
from .tools import (
    clean_nulls_tool,
    encode_categorical_tool,
    detect_schema_tool
)

logger = logging.getLogger(__name__)


class DataPreprocessorAgent(SpecialistAgent):
    """SpecialistAgent for tabular data preprocessing pipeline."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        name: str = "data_preprocessor",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name=name, agent_type=AgentType.SPECIALIST, config=config)
        self.llm = llm
        self.set_domain_expertise([
            "data_cleaning", "data_type_detection", "cardinality_analysis",
            "null_handling", "categorical_encoding", "schema_detection"
        ])
        # Store DataFrame in memory for reuse across steps
        self.df: Optional[pd.DataFrame] = None
        self.df_redis_key: Optional[str] = None
        self._data_fetched: bool = False

    def get_required_state_keys(self):
        """Return required state keys for preprocessing."""
        return ["db_id"]

    def _register_specialist_tools(self) -> None:
        """Register preprocessing tools."""
        # Tools are used directly as functions, not through registry
        # This method is kept for consistency with SpecialistAgent pattern
        pass

    def step(self, state: AgentState) -> AgentState:
        """Execute preprocessing step based on current_substep."""
        substep = state.get("current_substep", "full_pipeline")
        
        logger.info(f"DataPreprocessorAgent executing substep: {substep}")
        
        # Fetch data on first step if not already fetched
        if not self._data_fetched and substep != "fetch":
            state = self._fetch(state)
            if state.get("error"):
                return state
        
        try:
            if substep == "fetch":
                return self._fetch(state)
            elif substep == "schema_detection":
                return self._schema_detection(state)
            elif substep == "clean_nulls":
                return self._clean_nulls(state)
            elif substep == "encode":
                return self._encode(state)
            elif substep == "full_pipeline":
                return self._full_pipeline(state)
            else:
                raise ValueError(f"Unknown substep: {substep}")
        except Exception as e:
            logger.error(f"DataPreprocessorAgent substep {substep} failed: {str(e)}")
            state["error"] = f"Data preprocessing {substep} failed: {str(e)}"
            state.setdefault("warnings", []).append(str(e))
            return state

    def _fetch(self, state: AgentState) -> AgentState:
        """Fetch data from Redis or SQL and store in agent instance."""
        try:
            # Get parameters
            db_id = state.get("db_id")
            if not db_id:
                raise ValueError("Missing 'db_id' in state")

            final_sql = state.get("final_sql") or state.get("sql_query")
            df_raw_key = state.get("df_redis_key") or self.df_redis_key
            session_id = state.get("session_id", "default_session")
            force_refresh = bool(state.get("force_refresh", False))

            # Initialize metadata
            metadata = {
                "df_redis_key": None,
                "df_shape": None,
                "columns": None,
                "df_cached": False,
                "warnings": []
            }

            # Try to load from Redis cache first
            if df_raw_key and not force_refresh:
                try:
                    self.df = load_df_parquet(df_raw_key)
                    if self.df is not None:
                        metadata["df_redis_key"] = df_raw_key
                        metadata["df_cached"] = True
                except Exception as e:
                    metadata["warnings"].append(f"Failed to load cached dataframe: {e}")
                    self.df = None

            # If not loaded from cache, execute SQL
            if self.df is None:
                if not final_sql:
                    raise ValueError("No SQL query found and no cached data available. Provide 'final_sql' or 'df_redis_key'.")

                try:
                    database = Database()
                    rows, columns = database.run_query(sql=final_sql, db_id=db_id)
                    self.df = pd.DataFrame(rows, columns=columns)
                except Exception as e:
                    raise RuntimeError(f"Failed to execute SQL query: {e}") from e

                if self.df is None or self.df.empty:
                    metadata["warnings"].append("Query returned no rows.")

                # Save to Redis if session_id is provided
                if session_id:
                    redis_key = f"{db_id}:raw_df:{session_id}"
                    try:
                        save_df_parquet(redis_key, self.df)
                        metadata["df_redis_key"] = redis_key
                        metadata["df_cached"] = True
                    except Exception as e:
                        metadata["warnings"].append(f"Failed to cache dataframe in Redis: {e}")

            # Store in agent instance
            if self.df is not None:
                self.df_redis_key = metadata.get("df_redis_key")
                self._data_fetched = True
                metadata["df_shape"] = tuple(self.df.shape)
                metadata["columns"] = list(self.df.columns)
                
                # Update state with metadata
                state.update(metadata)
                if self.df_redis_key:
                    state["df_redis_key"] = self.df_redis_key

            return state

        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            state["error"] = f"Fetch failed: {str(e)}"
            return state

    def _schema_detection(self, state: AgentState) -> AgentState:
        """Detect data types and calculate cardinality."""
        try:
            # Use cached DataFrame from agent instance
            if self.df is None or self.df.empty:
                raise ValueError("No data available for schema detection. Run fetch first.")

            # Get configuration
            high_cardinality_threshold = state.get("high_cardinality_threshold", 50)

            # Use detect_schema_tool directly
            schema = detect_schema_tool(self.df, high_cardinality_threshold=high_cardinality_threshold)

            # Store schema in state
            state["variable_schema"] = schema
            state["variable_info"] = schema  # Backward compatibility

            # Check if HITL is needed
            warnings = []
            if schema.get("mixed_data_types"):
                warnings.append("Mixed data types detected (both continuous and categorical variables)")
            if schema.get("high_cardinality_vars"):
                warnings.append(
                    f"High cardinality variables detected: {', '.join(schema['high_cardinality_vars'][:5])}"
                )

            # Trigger HITL if needed
            hitl_required = (
                state.get("hitl_required", False) or
                schema.get("mixed_data_types") or
                len(schema.get("high_cardinality_vars", [])) > 0
            )

            if hitl_required and state.get("interactive", True):
                from langgraph.types import interrupt

                payload = {
                    "question": "Review detected data types and cardinality",
                    "schema": schema,
                    "warnings": warnings,
                    "required_fields": ["variable_schema"],
                    "hint": "Review and approve schema, or edit variable types/cardinality. "
                           "Set 'variable_schema' in response to override detected schema."
                }

                try:
                    user_input = interrupt(payload)
                    if user_input and isinstance(user_input, dict):
                        # Apply user edits if provided
                        if "variable_schema" in user_input:
                            state["variable_schema"] = user_input["variable_schema"]
                            state["variable_info"] = user_input["variable_schema"]
                        if "hitl_executed" not in user_input:
                            user_input["hitl_executed"] = True
                        state.update(user_input)
                except Exception as e:
                    logger.warning(f"HITL interrupt failed: {e}, continuing with detected schema")

            return state

        except Exception as e:
            logger.error(f"Schema detection failed: {e}")
            state["error"] = f"Schema detection failed: {str(e)}"
            return state

    def _clean_nulls(self, state: AgentState) -> AgentState:
        """Clean null values from DataFrame."""
        try:
            # Use cached DataFrame from agent instance
            if self.df is None or self.df.empty:
                raise ValueError("No data available for null cleaning. Run fetch first.")

            # Get configuration
            null_ratio = state.get("clean_nulls_ratio", 0.95)

            # Use clean_nulls_tool directly
            self.df, dropped_cols = clean_nulls_tool(self.df, null_ratio=null_ratio)

            # Save cleaned DataFrame to Redis
            if self.df_redis_key:
                from utils.redis_df import save_df_parquet
                processed_key = f"{self.df_redis_key}:clean_nulls"
                save_df_parquet(processed_key, self.df)
                self.df_redis_key = processed_key
                state["df_redis_key"] = self.df_redis_key
                state["df_shape"] = tuple(self.df.shape)
                state["columns"] = list(self.df.columns)

            state["dropped_null_columns"] = dropped_cols
            state.setdefault("warnings", []).extend(
                [f"Dropped column {col} due to high null ratio" for col in dropped_cols]
            )

            return state

        except Exception as e:
            logger.error(f"Clean nulls failed: {e}")
            state["error"] = f"Clean nulls failed: {str(e)}"
            return state

    def _encode(self, state: AgentState) -> AgentState:
        """Encode categorical variables."""
        try:
            # Use cached DataFrame from agent instance
            if self.df is None or self.df.empty:
                raise ValueError("No data available for encoding. Run fetch first.")

            # Get configuration
            threshold = state.get("one_hot_threshold", 20)

            # Use encode_categorical_tool directly
            self.df, encoded_cols = encode_categorical_tool(self.df, threshold=threshold)

            # Save encoded DataFrame to Redis
            if self.df_redis_key:
                from utils.redis_df import save_df_parquet
                processed_key = f"{self.df_redis_key}:encode"
                save_df_parquet(processed_key, self.df)
                self.df_redis_key = processed_key
                state["df_redis_key"] = self.df_redis_key
                state["df_shape"] = tuple(self.df.shape)
                state["columns"] = list(self.df.columns)

            state["encoded_columns"] = encoded_cols

            return state

        except Exception as e:
            logger.error(f"Encode failed: {e}")
            state["error"] = f"Encode failed: {str(e)}"
            return state

    def _full_pipeline(self, state: AgentState) -> AgentState:
        """Run full preprocessing pipeline."""
        # Run steps sequentially
        state = self._fetch(state)
        if state.get("error"):
            return state

        state = self._schema_detection(state)
        if state.get("error"):
            return state

        state = self._clean_nulls(state)
        if state.get("error"):
            return state

        # Skip encoding if flag is set (for causal discovery with mixed data)
        skip_encoding = state.get("skip_one_hot_encoding", False)
        if skip_encoding:
            logger.info("One-hot encoding skipped. Original data preserved for causal discovery algorithms.")
            state["encoded_columns"] = []
        else:
            state = self._encode(state)
            if state.get("error"):
                return state

        # Mark as completed
        state["data_preprocessing_completed"] = True

        return state

