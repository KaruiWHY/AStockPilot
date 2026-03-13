# -*- coding: utf-8 -*-
"""测试财报分析工具"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from tools import financial_tools

print("=" * 50)
print("Testing profitability indicators...")
print("=" * 50)
result = financial_tools.get_profitability_indicators('600000')
if 'error' in result:
    print('Error:', result['error'])
else:
    print('Report date:', result.get('report_date'))
    for ind in result.get('indicators', []):
        print(f"  {ind['name']}: {ind['value']}")
    print('ROE evaluation:', result.get('roe_evaluation', 'N/A'))

print("\n" + "=" * 50)
print("Testing growth indicators...")
print("=" * 50)
result = financial_tools.get_growth_indicators('600000')
if 'error' in result:
    print('Error:', result['error'])
else:
    print('Report date:', result.get('report_date'))
    for ind in result.get('indicators', []):
        print(f"  {ind['name']}: {ind['value']}")

print("\n" + "=" * 50)
print("Testing comprehensive financial analysis...")
print("=" * 50)
result = financial_tools.analyze_financial_report('600000')
if 'error' in result:
    print('Error:', result['error'])
else:
    print('Score:', result.get('overall_evaluation', {}).get('score', 'N/A'))
    print('Rating:', result.get('overall_evaluation', {}).get('rating', 'N/A'))
    print('Summary:', result.get('overall_evaluation', {}).get('summary', 'N/A'))

print("\nTest completed!")
