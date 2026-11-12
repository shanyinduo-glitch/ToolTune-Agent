"""Local ToolTune workbench: deterministic trace demo with optional model artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
import sys
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'src'))
from tooltune.environment import OrderToolEnvironment, ProductToolEnvironment

st.set_page_config(page_title='ToolTune-Agent Workbench', layout='wide')
st.title('ToolTune-Agent Workbench')
st.caption('可执行工具调用、状态变化与错误恢复演示')
left,right=st.columns([1,1.5])
with left:
    mode=st.selectbox('任务', ['订单查询','订单取消确认','商品搜索并查库存'])
    if mode=='订单查询': prompt=st.text_input('请求','查询订单 A2101 的状态')
    elif mode=='订单取消确认': prompt=st.text_input('请求','我想取消订单 A2101')
    else: prompt=st.text_input('请求','查找无线键盘并查看库存')
    run=st.button('执行确定性演示', type='primary')
    st.info('当前演示使用项目内确定性工具环境，模型适配器和完整推理结果保留在 outputs/adapters 与 outputs/metrics。')
if run:
    if mode=='订单查询':
        env=OrderToolEnvironment(); trace=[('get_order',{'order_id':'A2101'})];
    elif mode=='订单取消确认':
        env=OrderToolEnvironment(); trace=[('cancel_order',{'order_id':'A2101','confirmed':False}),('cancel_order',{'order_id':'A2101','confirmed':True})]
    else:
        env=ProductToolEnvironment(); trace=[('search_product',{'query':'keyboard'}),('check_inventory',{'product_id':'keyboard'})]
    results=[]
    for name,args in trace: results.append({'tool':name,'arguments':args,'result':env.execute(name,args)})
    with right:
        st.subheader('执行轨迹')
        for i,item in enumerate(results,1):
            st.write(f"步骤 {i}: `{item['tool']}`")
            st.json(item)
        st.subheader('最终状态')
        st.json(env.orders if isinstance(env,OrderToolEnvironment) else env.products)
else:
    with right:
        st.subheader('项目结果')
        report=ROOT/'outputs/reports/final_report.md'
        if report.exists(): st.markdown(report.read_text(encoding='utf-8'))
