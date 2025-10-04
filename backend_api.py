"""
COTCAgent Backend API
提供RESTful API接口，连接前端和COTCAgent核心功能
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import asyncio
import json
import os
from cotc_agent import COTCAgent, DeepSeekConfig

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储agent实例
agent = None

def initialize_agent():
    """初始化COTCAgent"""
    global agent
    if agent is None:
        config = DeepSeekConfig(
            api_key='sk-687c00f17caa45eaaa9756e96f49f6dc',
            api_base="https://api.deepseek.com/v1/chat/completions",
            model="deepseek-chat",
            max_tokens=2000,
            temperature=0.7,
            timeout=180,
            save_temp_files=True
        )
        agent = COTCAgent(config)
    return agent

@app.route('/')
def index():
    """主页 - 返回前端页面"""
    return render_template('web_interface.html')

@app.route('/api/patient/info', methods=['GET'])
def get_patient_info():
    """获取患者信息"""
    try:
        # 加载患者数据
        with open('patient_data/patient_0001.json', 'r', encoding='utf-8') as f:
            patient_data = json.load(f)
        
        return jsonify({
            'success': True,
            'data': {
                'patient_id': patient_data['patient_info']['id'],
                'total_indicators': patient_data['patient_info']['total_indicators'],
                'existing_diseases': len(patient_data['patient_info']['diseases']),
                'diseases': patient_data['patient_info']['diseases']
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analysis/query', methods=['POST'])
def analyze_query():
    """分析用户查询"""
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({
                'success': False,
                'error': '查询内容不能为空'
            }), 400
        
        # 初始化agent
        agent = initialize_agent()
        
        # 加载患者数据
        with open('patient_data/patient_0001.json', 'r', encoding='utf-8') as f:
            patient_data = json.load(f)
        
        # 异步处理查询
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                agent.process_user_query(user_query, patient_data)
            )
        finally:
            loop.close()
        
        # 格式化返回结果
        formatted_result = {
            'success': True,
            'data': {
                'temporal_analysis': result.get('temporal_analysis', {}),
                'detailed_analysis': result.get('detailed_analysis', {}),
                'disease_risks': [
                    {
                        'disease_id': risk.disease_id,
                        'disease_name': risk.disease_name,
                        'risk_score': risk.risk_score,
                        'confidence': risk.confidence,
                        'matched_symptoms': risk.matched_symptoms,
                        'missing_symptoms': risk.missing_symptoms
                    }
                    for risk in result.get('disease_risks', [])
                ],
                'inquiry_questions': result.get('active_inquiry_questions', []),
                'comprehensive_analysis': result.get('comprehensive_analysis', {})
            }
        }
        
        return jsonify(formatted_result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'分析过程中出现错误: {str(e)}'
        }), 500

@app.route('/api/analysis/status', methods=['GET'])
def get_analysis_status():
    """获取分析状态"""
    return jsonify({
        'success': True,
        'data': {
            'status': 'ready',
            'agent_initialized': agent is not None,
            'api_connected': True
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'service': 'COTCAgent API',
        'version': '1.0.0'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': '接口不存在'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': '服务器内部错误'
    }), 500

if __name__ == '__main__':
    print("启动COTCAgent API服务器...")
    print("访问地址: http://localhost:5000")
    print("API文档: http://localhost:5000/api/health")
    
    # 创建templates目录
    os.makedirs('templates', exist_ok=True)
    
    # 将HTML文件移动到templates目录
    if os.path.exists('web_interface.html'):
        import shutil
        shutil.move('web_interface.html', 'templates/web_interface.html')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
