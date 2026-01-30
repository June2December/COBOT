import os
import re

from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

from cobot2.MicController import MicController, MicConfig

from cobot2.wakeup_word import WakeupWord
from cobot2.stt import STT


############ GetKeyword Node ############
class GetKeyword():
    """
    expected(정답 답어) + stt_text(거수자 발화) 를 넣으면
    verdict/score/extracted/reason 을 반환하는 "검증기" 클래스
    """
    def __init__(self, temperature: float = 0.0):
        pkg = get_package_share_directory("cobot2")
        env_path = os.path.join(pkg, "resource", ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found. (resource/.env 로드 실패 또는 환경변수 미설정)")


        # temperature : 출력의 랜덤성(확률 분포의 퍼짐 정도)
        # 0 ~ 0.3 가 정석
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.3, openai_api_key=api_key
        )

        prompt_content = """
        당신은 음성인식(STT) 텍스트에서 "답어"를 검증하는 심사관입니다.

        [절대 규칙: 억지 매칭 금지]
        - stt_text에 expected가 "우연히" 포함될 가능성을 항상 고려하십시오.
        - 사용자가 단순 잡담/감탄/문장 일부로 말한 것을 "답어 발화"로 강제 해석하면 안 됩니다.
        - expected를 맞추기 위해 stt_text를 재구성하거나, 의미를 바꾸거나, 없는 단어를 만들어내면 안 됩니다.

        [OK 판정 조건(모두 만족에 가깝게)]
        1) expected(답어)가 stt_text에서 "독립적으로 발화"된 것으로 보일 것.
        - 예: 단독 발화("우졸리"), 또는 짧은 응답("우졸리입니다", "우졸리요", "우 졸 리")
        - 허용되는 변형: 공백 분리, 조사/어미 소량, 1~2글자 수준의 오탈자
        2) stt_text가 '답을 말하는 상황'으로 자연스러울 것.
        - 예: "정답은 우졸리", "우졸리 입니다", "우졸리요"
        - 또는 전체가 매우 짧은 발화(예: 1~4어절)인데 그 핵심이 expected일 것.
        3) 다음과 같은 경우는 OK로 하면 안 됩니다:
        - 긴 문장 속 일부에 우연히 섞인 경우
        - 감탄/잡담("아우 졸리다")처럼 expected와 무관한 의미인데 글자만 비슷한 경우
        - expected를 만들기 위해 단어 경계를 억지로 끊거나 합치는 경우(의도적 발화 근거가 약하면 UNKNOWN)

        [판정]
        - OK: 의도적으로 expected를 답어로 말한 것이 강하게 확실
        - MISMATCH: 다른 답어를 말했거나 expected가 명확히 아님
        - UNKNOWN: 애매함(억지 매칭 가능성이 있어 확정하면 위험)

        [출력 형식]
        한 줄로만 출력:
        <VERDICT> / extracted=<stt_text에서 답어로 보이는 원문 조각 또는 빈칸> / reason=<짧게 근거> / score=<0.00~1.00>

        [입력]
        expected: "{expected}"
        stt_text: "{stt_text}"
        """
        self.prompt_template = PromptTemplate(
            input_variables=["expected", "stt_text"], template=prompt_content
        )
        self.lang_chain = self.prompt_template | self.llm
        # self.lang_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def verify(self, expected: str, stt_text: str) -> dict:
        """
        return:
          {
            "verdict": "OK|MISMATCH|UNKNOWN",
            "extracted": str,
            "reason": str,
            "score": float,
            "raw": str
          }
        """
        expected = (expected or "").strip()
        stt_text = (stt_text or "").strip()

        raw = self.lang_chain.invoke({"expected": expected, "stt_text": stt_text}).content.strip()

        out = {
            "verdict": "UNKNOWN",
            "extracted": "",
            "reason": "",
            "score": 0.0,
            "raw": raw,
        }

        # 예: "OK / extracted=우 졸 리 / reason=... / score=0.93"
        m = re.match(
            r"^(OK|MISMATCH|UNKNOWN)\s*/\s*extracted=(.*?)\s*/\s*reason=(.*?)\s*/\s*score=([0-9]*\.?[0-9]+)\s*$",
            raw
        )
        if not m:
            # LLM 출력이 형식을 깨면 UNKNOWN 처리(안전)
            out["reason"] = "LLM output format invalid"
            return out

        out["verdict"] = m.group(1).strip()
        out["extracted"] = m.group(2).strip()
        out["reason"] = m.group(3).strip()
        try:
            out["score"] = float(m.group(4))
        except ValueError:
            out["score"] = 0.0
        return out


def main():
    verifier = GetKeyword()
    expected = "아졸리"

    tests = [
        "아졸리",          # OK 기대
        "아 졸 리",        # OK 기대
        "아졸리입니다",     # OK 기대
        "아 졸리다",       # MISMATCH/UNKNOWN 기대 (억지매칭 금지)
        "졸리다",          # MISMATCH 기대
        "아우 졸리다",     # MISMATCH 기대
    ]

    for stt_text in tests:
        r = verifier.verify(expected=expected, stt_text=stt_text)
        print(f"\nexpected={expected} | stt_text={stt_text}")
        print(r)


if __name__ == "__main__":
    main()