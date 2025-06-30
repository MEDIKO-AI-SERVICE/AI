import requests
import xml.etree.ElementTree as ET
import configparser
import time
from typing import List, Dict, Any

class DrugAPIClient:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('keys.config')
        self.api_key = config['API_KEYS']['public_portal_api_key']
        self.base_url = "http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
        
    def get_all_drugs(self) -> List[Dict[str, Any]]:
        all_drugs = []
        page_no = 1
        num_of_rows = 100  # 한 번에 가져올 데이터 수
        
        while True:
            try:
                params = {
                    'serviceKey': self.api_key,
                    'pageNo': page_no,
                    'numOfRows': num_of_rows,
                    'type': 'xml'
                }
                
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                
                # XML 파싱
                root = ET.fromstring(response.content)
                
                # totalCount 확인
                total_count = int(root.find('.//totalCount').text)
                print(f"Total drugs available: {total_count}")
                
                # items 파싱
                items = root.findall('.//item')
                if not items:
                    break
                    
                for item in items:
                    drug_info = {
                        'entpName': self._get_text(item, 'entpName'),
                        'itemName': self._get_text(item, 'itemName'),
                        'itemSeq': self._get_text(item, 'itemSeq'),
                        'efcyQesitm': self._get_text(item, 'efcyQesitm'),
                        'useMethodQesitm': self._get_text(item, 'useMethodQesitm'),
                        'atpnWarnQesitm': self._get_text(item, 'atpnWarnQesitm'),
                        'atpnQesitm': self._get_text(item, 'atpnQesitm'),
                        'intrcQesitm': self._get_text(item, 'intrcQesitm'),
                        'seQesitm': self._get_text(item, 'seQesitm'),
                        'depositMethodQesitm': self._get_text(item, 'depositMethodQesitm'),
                        'openDe': self._get_text(item, 'openDe'),
                        'updateDe': self._get_text(item, 'updateDe'),
                        'itemImage': self._get_text(item, 'itemImage'),
                        'bizrno': self._get_text(item, 'bizrno')
                    }
                    all_drugs.append(drug_info)
                
                print(f"Collected {len(all_drugs)} drugs so far...")
                
                # 다음 페이지로
                page_no += 1
                
                # API 호출 제한을 위한 딜레이
                time.sleep(0.1)
                
                # 모든 데이터를 수집했는지 확인
                if len(all_drugs) >= total_count:
                    break
                    
            except Exception as e:
                print(f"Error collecting drugs from page {page_no}: {e}")
                break
        
        print(f"Total drugs collected: {len(all_drugs)}")
        return all_drugs
    
    def _get_text(self, element, tag_name: str) -> str:
        tag = element.find(tag_name)
        return tag.text if tag is not None else "" 