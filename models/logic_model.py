import json
import os
from typing import Dict, List, Any
from utils.openai_helper import OpenAIHelper

class LogicModel:
    """
    The Logic Model focuses on rational analysis, efficiency, and practical outcomes.
    It evaluates decisions based on logical reasoning, cost-benefit analysis, and optimal resource allocation.
    """
    
    def __init__(self):
        self.openai_helper = OpenAIHelper()
        self.logical_frameworks = [
            "Cost-benefit analysis",
            "Risk assessment",
            "Efficiency optimization",
            "Resource allocation",
            "Statistical analysis",
            "Logical reasoning",
            "Process improvement",
            "Performance metrics"
        ]
        self.knowledge_file = "data/logic_knowledge.json"
        self.load_knowledge()
    
    def load_knowledge(self):
        """Load logical analysis knowledge and case studies"""
        try:
            with open(self.knowledge_file, 'r') as f:
                self.knowledge = json.load(f)
        except FileNotFoundError:
            self.knowledge = {
                "analysis_cases": [],
                "logical_frameworks": self.logical_frameworks,
                "optimization_strategies": [],
                "performance_metrics": []
            }
            self.save_knowledge()
    
    def save_knowledge(self):
        """Save logical knowledge to file"""
        os.makedirs(os.path.dirname(self.knowledge_file), exist_ok=True)
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.knowledge, f, indent=2)
    
    def analyze_logical_efficiency(self, scenario: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze a scenario for logical efficiency and optimal outcomes
        """
        try:
            prompt = f"""
            As a logical analyst focused on efficiency and optimal outcomes,
            analyze the following scenario:
            
            Scenario: {scenario}
            Context: {context}
            
            Apply these logical frameworks:
            {', '.join(self.logical_frameworks)}
            
            Provide analysis in JSON format with:
            - efficiency_score: number from 1-10 (10 being most efficient)
            - cost_benefit_ratio: string describing cost vs benefit
            - risks: list of potential risks and their probability
            - optimization_opportunities: list of ways to improve efficiency
            - resource_requirements: list of resources needed
            - expected_outcomes: list of likely results
            - success_probability: number from 0-100 (percentage)
            """
            
            response = self.openai_helper.get_structured_response(prompt)
            
            # Store analysis for learning
            self.knowledge["analysis_cases"].append({
                "scenario": scenario,
                "context": context,
                "analysis": response,
                "timestamp": self.openai_helper.get_current_timestamp()
            })
            self.save_knowledge()
            
            return response
            
        except Exception as e:
            return {
                "efficiency_score": 5,
                "cost_benefit_ratio": f"Unable to analyze: {str(e)}",
                "risks": ["Analysis uncertainty"],
                "optimization_opportunities": ["Gather more data"],
                "resource_requirements": ["Unknown"],
                "expected_outcomes": ["Uncertain"],
                "success_probability": 50
            }
    
    def provide_logical_solution(self, problem: str, constraints: str = "") -> str:
        """
        Provide a logical, data-driven solution to a problem
        """
        try:
            # Include relevant past analyses
            relevant_cases = self.get_relevant_logic_cases(problem)
            cases_context = ""
            if relevant_cases:
                cases_context = "\n\nRelevant past analyses:\n"
                for case in relevant_cases[-3:]:
                    cases_context += f"- Efficiency: {case['analysis'].get('efficiency_score', 'N/A')}/10\n"
                    cases_context += f"  Optimization: {', '.join(case['analysis'].get('optimization_opportunities', [])[:2])}\n"
            
            prompt = f"""
            As a logical problem solver focused on efficiency and optimal outcomes,
            provide a solution for this problem:
            
            Problem: {problem}
            Constraints: {constraints}
            {cases_context}
            
            Logical Frameworks to apply:
            {', '.join(self.logical_frameworks)}
            
            Please provide:
            1. A step-by-step logical approach
            2. Cost-benefit analysis
            3. Risk mitigation strategies
            4. Measurable success criteria
            5. Resource optimization suggestions
            6. Alternative solutions ranked by efficiency
            
            Focus on practicality, efficiency, and measurable outcomes.
            """
            
            response = self.openai_helper.get_text_response(prompt)
            return response
            
        except Exception as e:
            return f"I'm having difficulty analyzing this logically right now: {str(e)}. I recommend breaking down the problem into smaller components and gathering more data for analysis."
    
    def get_relevant_logic_cases(self, query: str) -> List[Dict]:
        """
        Find logic cases relevant to the current query
        """
        try:
            relevant_cases = []
            query_lower = query.lower()
            
            for case in self.knowledge.get("analysis_cases", []):
                scenario_text = case.get("scenario", "").lower()
                if any(word in scenario_text for word in query_lower.split()):
                    relevant_cases.append(case)
            
            return relevant_cases[-5:]
            
        except Exception:
            return []
    
    def optimize_resource_allocation(self, resources: List[str], objectives: List[str], constraints: List[str] = None) -> Dict[str, Any]:
        """
        Provide optimal resource allocation recommendations
        """
        try:
            constraints_str = ', '.join(constraints) if constraints else "None specified"
            
            prompt = f"""
            Optimize resource allocation for maximum efficiency:
            
            Available Resources: {', '.join(resources)}
            Objectives: {', '.join(objectives)}
            Constraints: {constraints_str}
            
            Provide optimization in JSON format with:
            - allocation_strategy: string describing optimal allocation
            - efficiency_gain: number from 0-100 (percentage improvement)
            - priority_ranking: list of objectives ranked by importance
            - resource_utilization: object mapping resources to recommended usage
            - bottlenecks: list of potential limiting factors
            - success_metrics: list of ways to measure success
            """
            
            response = self.openai_helper.get_structured_response(prompt)
            
            # Store optimization strategy
            self.knowledge["optimization_strategies"].append({
                "resources": resources,
                "objectives": objectives,
                "constraints": constraints,
                "strategy": response,
                "timestamp": self.openai_helper.get_current_timestamp()
            })
            self.save_knowledge()
            
            return response
            
        except Exception as e:
            return {
                "allocation_strategy": f"Unable to optimize: {str(e)}",
                "efficiency_gain": 0,
                "priority_ranking": objectives,
                "resource_utilization": {},
                "bottlenecks": ["Analysis limitation"],
                "success_metrics": ["Manual evaluation needed"]
            }
    
    def calculate_roi_analysis(self, investment: str, expected_returns: str, timeframe: str) -> Dict[str, Any]:
        """
        Calculate return on investment and provide logical recommendations
        """
        try:
            prompt = f"""
            Perform ROI analysis for this investment:
            
            Investment: {investment}
            Expected Returns: {expected_returns}
            Timeframe: {timeframe}
            
            Provide analysis in JSON format with:
            - roi_percentage: estimated ROI as percentage
            - payback_period: estimated time to break even
            - risk_level: string (Low/Medium/High)
            - logical_recommendation: string with recommendation
            - key_assumptions: list of assumptions made
            - sensitivity_factors: list of factors that could affect ROI
            """
            
            response = self.openai_helper.get_structured_response(prompt)
            
            # Store ROI analysis
            self.knowledge["performance_metrics"].append({
                "investment": investment,
                "analysis": response,
                "timestamp": self.openai_helper.get_current_timestamp()
            })
            self.save_knowledge()
            
            return response
            
        except Exception as e:
            return {
                "roi_percentage": "Unable to calculate",
                "payback_period": "Unknown",
                "risk_level": "High (due to analysis uncertainty)",
                "logical_recommendation": f"Gather more data: {str(e)}",
                "key_assumptions": ["Analysis incomplete"],
                "sensitivity_factors": ["All factors uncertain"]
            }
    
    def add_logical_framework(self, framework: str):
        """
        Add a new logical framework to consider
        """
        if framework not in self.logical_frameworks:
            self.logical_frameworks.append(framework)
            self.knowledge["logical_frameworks"] = self.logical_frameworks
            self.save_knowledge()
    
    def get_logic_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the logic model's knowledge and capabilities
        """
        return {
            "logical_frameworks": len(self.logical_frameworks),
            "cases_analyzed": len(self.knowledge.get("analysis_cases", [])),
            "optimizations_performed": len(self.knowledge.get("optimization_strategies", [])),
            "roi_analyses": len(self.knowledge.get("performance_metrics", [])),
            "focus_areas": [
                "Efficiency optimization",
                "Cost-benefit analysis",
                "Risk assessment",
                "Resource allocation",
                "Performance measurement"
            ]
        }
