东盟医疗人工智能本地化验证注册库

用途范围：Repository.register_scope 记录人群和用途，link_cohort 将人群证据
关联范围；issue_scope 签发时固定队列版本和指纹，证据在签发前改变须重新关联。
restrict_scope 在已签发记录之上追加后续限制意见，不覆盖既有结论。scope_history
可查范围的签发及限制历史。cohort_guard 提供基础 Wilson 统计和同意范围判断。

运行验收：
python3 -m clinic_localizer run --fixture fixtures/acceptance.json --database /tmp/clinic_localizer.sqlite3 --output /tmp/clinic_localizer-output

核验：
python3 -m clinic_localizer verify --database /tmp/clinic_localizer.sqlite3 --output /tmp/clinic_localizer-output

全量测试：
python3 -m unittest discover -s tests -v

按模块测试：
python3 -m unittest tests.cohorts.test_protocol -v

构建检查：
python3 -m compileall -q clinic_localizer tests

两阶段验证（候选选择 / 独立确认）：
statistics/two_stage_metrics.py 为纯领域逻辑，governance/two_stage.py 为持久化编排，
Repository 暴露 create_cycle / assign_split / record_candidate / freeze_selection /
submit_confirmation / issue_acceptance / restrict_acceptance / basis_drift_report /
cycle_history。关键边界：
- assign_split 强制选择集与确认集按受试者标识和机构双不相交，重叠直接 ContractError；
- record_candidate 只接受 selection 队列；freeze_selection 后状态锁定，确认数据上无调参入口；
- submit_confirmation 只评估冻结的 model_id + operating_threshold，新补报走新 review 并作废旧版；
- 指标给出漏诊率（1-敏感度）、误报率（1-特异度）的 Wilson 保守上界，并按人群/机构分层比较；
- issue_acceptance 签发前重新核对记录指纹与 validation 同意，漂移或撤回则拒绝签发；
  小样本/高缺失/单机构仅在 allow_limited=True 时作限制接受，supported_uses 列明仍支持的用途；
  结局缺失或安全底线不过直接报错，不落任何接受记录；
- acceptance_issued 之后补报/撤回不改变原时点结论，只经 restrict_acceptance 追加限制，
  cycle_history 与 basis_drift_report 可查依据、漂移与全部后续限制。
对应测试：python3 -m unittest tests.stats.test_two_stage_metrics tests.governance.test_two_stage -v
