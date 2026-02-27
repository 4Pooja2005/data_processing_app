import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table, TableStyle
import os


class DataHealthScorecard:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.total_cells = self.df.shape[0] * self.df.shape[1]

    # -----------------------------
    # 1. Missing Value %
    # -----------------------------
    def missing_percentage(self):
        missing = self.df.isnull().sum().sum()
        return (missing / self.total_cells) * 100

    # -----------------------------
    # 2. Duplicate Rate
    # -----------------------------
    def duplicate_percentage(self):
        duplicates = self.df.duplicated().sum()
        return (duplicates / len(self.df)) * 100

    # -----------------------------
    # 3. Outlier Count (IQR method)
    # -----------------------------
    def outlier_count(self):
        numeric_cols = self.df.select_dtypes(include=np.number)
        total_outliers = 0

        for col in numeric_cols:
            Q1 = numeric_cols[col].quantile(0.25)
            Q3 = numeric_cols[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = numeric_cols[(numeric_cols[col] < lower) | 
                                    (numeric_cols[col] > upper)]
            total_outliers += outliers.shape[0]

        return total_outliers

    # -----------------------------
    # 4. Schema Consistency
    # -----------------------------
    def schema_consistency(self):
        inconsistent = 0

        for col in self.df.columns:
            dtype_counts = self.df[col].apply(type).value_counts()
            if len(dtype_counts) > 1:
                inconsistent += 1

        return (1 - inconsistent / len(self.df.columns)) * 100

    # -----------------------------
    # Dynamic Column Issues
    # -----------------------------
    def get_high_null_cols(self, threshold=0.5):
        nulls = self.df.isnull().sum() / len(self.df)
        return nulls[nulls > threshold].index.tolist()

    def get_constant_cols(self):
        # Columns with exactly 1 unique value
        return [col for col in self.df.columns if self.df[col].nunique(dropna=False) <= 1]

    def get_high_cardinality_cols(self, threshold=0.9):
        # Text/categorical columns where > 90% of rows are unique
        cat_cols = self.df.select_dtypes(include=['object', 'string']).columns
        issues = []
        for col in cat_cols:
            if self.df[col].nunique() / len(self.df) > threshold:
                issues.append(col)
        return issues

    # -----------------------------
    # 5. Health Score (Weighted)
    # -----------------------------
    def compute_health_score(self):
        missing = self.missing_percentage()
        duplicates = self.duplicate_percentage()
        outliers = self.outlier_count()
        schema = self.schema_consistency()

        score = 100
        score -= missing * 0.4
        score -= duplicates * 0.2
        score -= (outliers / len(self.df)) * 10
        score = (score * 0.5) + (schema * 0.5)

        return max(0, round(score, 2))

    # -----------------------------
    # Generate DataFrame
    # -----------------------------
    def generate_report_df(self):
        missing = round(self.missing_percentage(), 2)
        duplicates = round(self.duplicate_percentage(), 2)
        outliers = self.outlier_count()
        schema = round(self.schema_consistency(), 2)
        score = self.compute_health_score()

        data = {
            "Metric": [
                "Missing %",
                "Duplicate %",
                "Outlier Count",
                "Schema Consistency %",
                "Final Health Score"
            ],
            "Value": [
                f"{missing}%",
                f"{duplicates}%",
                str(outliers),
                f"{schema}%",
                f"{score}/100"
            ]
        }
        return pd.DataFrame(data)

    # -----------------------------
    # Generate Charts
    # -----------------------------
    def generate_charts(self):
        missing = self.missing_percentage()
        duplicates = self.duplicate_percentage()
        outliers = self.outlier_count()

        plt.figure()
        metrics = ['Missing %', 'Duplicate %', 'Outliers']
        values = [missing, duplicates, outliers]

        plt.bar(metrics, values)
        plt.title("Data Health Metrics")
        chart_path = "health_chart.png"
        plt.savefig(chart_path)
        plt.close()

        return chart_path

    # -----------------------------
    # Generate AI Summary
    # -----------------------------
    def generate_ai_summary(self, api_key):
        if not api_key:
            return "AI Summary unavailable (No API Key provided)."
        
        try:
            from google import genai
            
            client = genai.Client(api_key=api_key)
            
            missing = round(self.missing_percentage(), 2)
            duplicates = round(self.duplicate_percentage(), 2)
            outliers = self.outlier_count()
            schema = round(self.schema_consistency(), 2)
            score = self.compute_health_score()
            
            high_nulls = self.get_high_null_cols()
            constants = self.get_constant_cols()
            cardinality = self.get_high_cardinality_cols()
            
            summary_stats = self.df.describe(include='all').to_string()
            
            prompt = f"""
            You are a Data Analyst. I have a dataset with the following health metrics:
            - Missing Values: {missing}%
            - Duplicate Rows: {duplicates}%
            - Outlier Count: {outliers}
            - Schema Consistency: {schema}%
            - Overall Health Score: {score}/100
            
            ### Specific Column Warnings
            - Columns missing >50% data: {high_nulls if high_nulls else 'None'}
            - Constant Columns (only 1 value): {constants if constants else 'None'}
            - High Cardinality Text Columns (likely IDs/noisy): {cardinality if cardinality else 'None'}
            
            Here is a brief statistical summary of the dataset:
            {summary_stats}
            
            Write a concise, 2-paragraph professional analysis of this dataset's health. 
            Highlight the specific column warnings if any exist and what the user should do about them (e.g. drop them, impute them).
            Keep it strictly professional, direct, and under 150 words. Do not use markdown formatting.
            """
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            
            return response.text
            
        except ImportError:
            return "AI Summary unavailable: google-genai library is not installed. Please run: pip install google-genai"
        except Exception as e:
            return f"AI Summary unavailable: Error connecting to Gemini API - {str(e)}"

    # -----------------------------
    # Generate PDF
    # -----------------------------
    def generate_pdf(self, filename="Data_Health_Report.pdf", api_key=None):
        doc = SimpleDocTemplate(filename)
        elements = []

        styles = getSampleStyleSheet()
        elements.append(Paragraph("Data Health Scorecard Report", styles['Title']))
        elements.append(Spacer(1, 0.3 * inch))

        # Generate AI Summary
        if api_key:
            elements.append(Paragraph("AI Analysis", styles['Heading2']))
            elements.append(Spacer(1, 0.2 * inch))
            
            ai_summary_text = self.generate_ai_summary(api_key)
            # Create a custom style for the summary box
            summary_style = ParagraphStyle(
                'AISummary',
                parent=styles['Normal'],
                backColor=colors.HexColor('#f0f4f8'),
                borderColor=colors.HexColor('#cbd5e1'),
                borderWidth=1,
                borderPadding=10,
                borderRadius=5,
                leading=16,
                spaceAfter=15
            )
            paragraphs = ai_summary_text.strip().split('\n\n')
            for p_text in paragraphs:
                clean_text = ' '.join(p_text.replace('\n', ' ').split())
                if clean_text:
                    elements.append(Paragraph(clean_text, summary_style))
                    elements.append(Spacer(1, 0.1 * inch))
            
            elements.append(Spacer(1, 0.2 * inch))

        missing = round(self.missing_percentage(), 2)
        duplicates = round(self.duplicate_percentage(), 2)
        outliers = self.outlier_count()
        schema = round(self.schema_consistency(), 2)
        score = self.compute_health_score()

        data = [
            ["Metric", "Value"],
            ["Missing %", f"{missing}%"],
            ["Duplicate %", f"{duplicates}%"],
            ["Outlier Count", str(outliers)],
            ["Schema Consistency %", f"{schema}%"],
            ["Final Health Score", f"{score}/100"]
        ]

        # Dynamically append specific warnings if they exist
        high_nulls = self.get_high_null_cols()
        if high_nulls:
            data.append(["High Nulls (>50%)", ", ".join(high_nulls)])
            
        constants = self.get_constant_cols()
        if constants:
            data.append(["Constant Cols", ", ".join(constants)])
            
        cardinality = self.get_high_cardinality_cols()
        if cardinality:
            data.append(["High Cardinality", ", ".join(cardinality)])

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.5 * inch))

        chart_path = self.generate_charts()
        elements.append(Image(chart_path, width=5*inch, height=3*inch))

        doc.build(elements)

        if os.path.exists(chart_path):
            os.remove(chart_path)

        return filename