import logging
import os
from typing import Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
WORKFLOW_NAME = "eda_workflow"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r") as f:
        return f.read()


class EDAWorkflow:
    """
    Exploratory Data Analysis workflow that performs consistent, first-pass analysis of datasets.
    
    Uses a fixed set of predefined analysis tools to produce structured, tabular outputs.
    Operates sequentially and deterministically through baseline EDA steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_workflow() is called.
    """
    
    def __init__(
        self,
        model=None,
        log=False,
        log_path=None,
        checkpointer: Optional[object] = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_eda_baseline_workflow(
            model=model,
            log=log,
            log_path=log_path,
            checkpointer=checkpointer
        )
    
    def invoke_workflow(self, filepath: str, **kwargs):
        """
        Run EDA analysis on the provided dataset.
        
        Parameters
        ----------
        filepath : str
            Path to the dataset file.
        **kwargs
            Additional arguments passed to the underlying graph invoke method.
        
        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        df = pd.read_csv(filepath)
        
        response = self._compiled_graph.invoke({
            "dataframe": df.to_dict(),
            "results": {},
            "observations": {},
            "current_step": "",
            "summary": "",
            "recommendations": [],
        }, **kwargs)
        
        self.response = response
        return None
    
    def get_summary(self):
        """Retrieves the analysis summary."""
        if self.response:
            return self.response.get("summary")
    
    def get_recommendations(self):
        """Retrieves the recommendations."""
        if self.response:
            return self.response.get("recommendations")
    
    def get_results(self):
        """Retrieves the full analysis results."""
        if self.response:
            return self.response.get("results")
    
    def get_observations(self):
        """Retrieves all observations from analysis steps."""
        if self.response:
            return self.response.get("observations")


def make_eda_baseline_workflow(
    model=None,
    log=False,
    log_path=None,
    checkpointer: Optional[object] = None
):
    """
    Factory function that creates a compiled LangGraph workflow for baseline EDA.
    
    Performs automated first-pass analysis with fixed analysis steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process EDA requests.
    """
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class EDAState(TypedDict):
        dataframe: dict
        results: dict
        observations: dict[str, list[str]]
        current_step: str
        summary: str
        recommendations: list[str]
    
    def profile_dataset_node(state: EDAState):
        """Generate dataset profile with basic statistics."""
        logger.info("Profiling dataset")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary": (
                df[numeric_cols].describe().to_dict() if numeric_cols else {}
            ),
            "categorical_summary": {
                col: df[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            },
        }
        
        results["profile_dataset"] = profile
        
        return {
            "current_step": "profile_dataset",
            "results": results,
        }
    
    def analyze_missingness_node(state: EDAState):
        """Analyze missing values in the dataset."""
        logger.info("Analyzing missingness")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        missing_count = df.isnull().sum().to_dict()
        missing_pct = (
            (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        )
        
        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 20}
        
        missingness = {
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "high_missing_columns": high_missing,
            "complete_rows": int(df.dropna().shape[0]),
            "complete_rows_pct": (
                round(df.dropna().shape[0] / len(df) * 100, 2)
                if len(df) > 0 else 0
            ),
        }
        
        results["analyze_missingness"] = missingness
        
        return {
            "current_step": "analyze_missingness",
            "results": results,
        }
    
    def compute_aggregates_node(state: EDAState):
        """Compute aggregates (mean, median, min, max, std) for each numeric column."""
        logger.info("Computing aggregates")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        aggregations = {}

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                aggregations[col] = {
                    "mean": None,
                    "median": None,
                    "min": None,
                    "max": None,
                    "std": None,
                }
                continue
            mean_val = series.mean()
            median_val = series.median()
            min_val = series.min()
            max_val = series.max()
            std_val = series.std()

            def _to_native(x):
                if pd.isna(x):
                    return None
                return float(x) if isinstance(x, (float, int)) else x

            aggregations[col] = {
                "mean": _to_native(mean_val),
                "median": _to_native(median_val),
                "min": _to_native(min_val),
                "max": _to_native(max_val),
                "std": _to_native(std_val),
            }

        results["aggregates_analysis"] = {"aggregations": aggregations}

        return {
            "current_step": "aggregates_analysis",
            "results": results,
        }
    
    def analyze_relationships_node(state: EDAState):
        """Analyze relationships: correlations with |r| >= 0.50, crosstabs at most 5x5."""
        logger.info("Analyzing relationships")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        def _to_native(x):
            if pd.isna(x):
                return None
            try:
                f = float(x)
                return int(f) if f == int(f) else f
            except (ValueError, TypeError):
                return x

        def _dict_to_native(d):
            if isinstance(d, dict):
                return {k: _dict_to_native(v) for k, v in d.items()}
            if isinstance(d, (list, tuple)):
                return [_dict_to_native(v) for v in d]
            return _to_native(d)

        # Numeric vs numeric: only correlations with |r| >= 0.50 (exclude diagonal)
        numeric_correlation = {}
        if len(numeric_cols) >= 2:
            corr_df = df[numeric_cols].corr()
            for i, c1 in enumerate(numeric_cols):
                for c2 in numeric_cols[i + 1 :]:  # upper triangle only, skip diagonal
                    r = corr_df.loc[c1, c2]
                    if pd.notna(r) and abs(r) >= 0.50:
                        numeric_correlation[f"{c1}_vs_{c2}"] = _to_native(r)

        # Categorical vs categorical: only crosstabs with at most 5x5
        categorical_crosstabs = {}
        for i, c1 in enumerate(categorical_cols):
            for c2 in categorical_cols[i + 1 :]:
                try:
                    ct = pd.crosstab(df[c1], df[c2], dropna=False)
                    if ct.shape[0] <= 5 and ct.shape[1] <= 5:
                        categorical_crosstabs[f"{c1}_vs_{c2}"] = _dict_to_native(ct.to_dict())
                except (KeyError, TypeError):
                    pass

        results["analyze_relationships"] = {
            "numeric_correlation": numeric_correlation,
            "categorical_crosstabs": categorical_crosstabs,
        }

        return {
            "current_step": "analyze_relationships",
            "results": results,
        }
    
    def extract_observations_node(state: EDAState):
        """Extract observations from the latest analysis results using LLM."""
        logger.info("Extracting observations")
        
        current_step = state.get("current_step", "")
        results = state.get("results", {})
        observations = state.get("observations", {})
        
        if model is None or not current_step or current_step not in results:
            return {"observations": observations}
        
        step_results = results.get(current_step, {})
        
        class ObservationOutput(BaseModel):
            observations: list[str] = Field(description="1-2 concise, actionable observations")
        
        observation_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("extract_observations_system.txt")),
            ("human", load_prompt("extract_observations_human.txt")),
        ])
        
        chain = observation_prompt | model.with_structured_output(ObservationOutput)
        response = chain.invoke({
            "step_name": current_step.replace("_", " ").title(),
            "results": str(step_results)
        })
        
        observations[current_step] = response.observations
        
        return {
            "observations": observations,
        }
    
    def synthesize_findings_node(state: EDAState):
        """Synthesize accumulated findings into summary and recommendations."""
        logger.info("Synthesizing findings")
        
        observations = state.get("observations", {})
        
        if model is None:
            return {
                "summary": "No LLM provided for synthesis",
                "recommendations": [],
            }
        
        class SynthesisOutput(BaseModel):
            summary: str = Field(description="A concise 2-3 sentence summary of key findings")
            recommendations: list[str] = Field(description="3-5 actionable recommendations")
        
        all_observations = []
        for step_name, step_obs in observations.items():
            all_observations.append(f"\n{step_name.replace('_', ' ').title()}:")
            for obs in step_obs:
                all_observations.append(f"  - {obs}")
        
        observations_text = "\n".join(all_observations)
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("synthesize_findings_system.txt")),
            ("human", load_prompt("synthesize_findings_human.txt")),
        ])
        
        chain = synthesis_prompt | model.with_structured_output(SynthesisOutput)
        response = chain.invoke({"observations": observations_text})
        
        return {
            "summary": response.summary,
            "recommendations": response.recommendations,
        }
    
    workflow = StateGraph(EDAState)
    
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("extract_observations_1", extract_observations_node)
    workflow.add_node("analyze_missingness", analyze_missingness_node)
    workflow.add_node("extract_observations_2", extract_observations_node)
    workflow.add_node("compute_aggregates", compute_aggregates_node)
    workflow.add_node("extract_observations_3", extract_observations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("extract_observations_4", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)
    
    workflow.set_entry_point("profile_dataset")
    
    workflow.add_edge("profile_dataset", "extract_observations_1")
    workflow.add_edge("extract_observations_1", "analyze_missingness")
    workflow.add_edge("analyze_missingness", "extract_observations_2")
    workflow.add_edge("extract_observations_2", "compute_aggregates")
    workflow.add_edge("compute_aggregates", "extract_observations_3")
    workflow.add_edge("extract_observations_3", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "extract_observations_4")
    workflow.add_edge("extract_observations_4", "synthesize_findings")
    workflow.add_edge("synthesize_findings", END)
    
    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)
    
    return app
