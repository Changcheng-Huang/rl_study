import streamlit as st

# =====================================================
# Common UI components for Theoretical Notebooks
# =====================================================


def page_header(title: str, subtitle: str):
    st.header(title)
    st.caption(subtitle)
    st.divider()


def right_card(title: str, bullets=None, body=None):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if bullets:
            st.markdown("\n".join([f"- {b}" for b in bullets]))
        if body:
            st.markdown(body)



def self_check(items: list[str]):
    st.markdown("#### Self-check")
    for it in items:
        st.checkbox(it, value=False)



def render_quiz(quiz_key: str, questions: list[dict]):
    """Multiple-choice quiz with submit + feedback.

    Each question dict:
      {
        "q": str,
        "options": [str, ...],
        "answer": int,  # index
        "explain": str
      }
    """

    st.markdown("#### Checkpoint Quiz")
    st.caption("Choose answers and click Submit to see results and explanations.")

    with st.form(key=f"quiz_form_{quiz_key}"):
        # collect answers inside a form so results appear only after submit
        for i, item in enumerate(questions, start=1):
            st.markdown(f"**Q{i}. {item['q']}**")
            st.radio(
                label=f"Question {i} answer",
                options=item["options"],
                index=None,
                key=f"{quiz_key}_q{i}",
                label_visibility="collapsed",
            )
            st.markdown("---")

        submitted = st.form_submit_button("Submit Answers", type="primary")

    if not submitted:
        return

    score = 0
    total = len(questions)

    st.subheader("Results")
    for i, item in enumerate(questions, start=1):
        correct_opt = item["options"][item["answer"]]
        user_choice = st.session_state.get(f"{quiz_key}_q{i}", None)
        ok = (user_choice == correct_opt)
        score += int(ok)

        if ok:
            st.success(f"Q{i}: Correct ✅")
        else:
            st.error(f"Q{i}: Incorrect ❌")

        st.markdown(f"**Your answer:** {user_choice if user_choice else 'No answer'}")
        st.markdown(f"**Correct answer:** {correct_opt}")
        with st.expander("Show explanation", expanded=False):
            st.write(item.get("explain", ""))
        st.markdown("---")

    st.metric("Score", f"{score}/{total}")
