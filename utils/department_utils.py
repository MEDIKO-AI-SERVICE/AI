def extract_korean_department(department_input):
    """
    진료과 입력에서 한국어 진료과명만 추출합니다.
    예: '신경과 (Singyeonggwa, ประสาทวิทยา)' -> '신경과'
    여러 진료과가 리스트로 들어올 경우: ['진료과1 (로마자, 번역)', '진료과2 (로마자, 번역)'] -> '진료과1, 진료과2'
    """
    if not department_input:
        return "내과"
    
    # 리스트인 경우 각 항목을 처리
    if isinstance(department_input, list):
        processed_departments = []
        for dept in department_input:
            processed_dept = extract_single_department(dept)
            if processed_dept:
                processed_departments.append(processed_dept)
        return ", ".join(processed_departments) if processed_departments else "내과"
    
    # 문자열인 경우 단일 처리
    return extract_single_department(department_input)

def extract_single_department(department_input):
    """
    단일 진료과 문자열에서 한국어 진료과명만 추출합니다.
    """
    if not department_input:
        return "내과"
    
    # 괄호가 있는 경우 (로마자, 번역) 형태
    if '(' in department_input and ')' in department_input:
        # 괄호 앞의 한국어 진료과명 추출
        korean_dept = department_input.split('(')[0].strip()
        return korean_dept
    
    # 괄호가 없는 경우 그대로 반환
    return department_input.strip() 