# BG001：限定主假设获支持，停止本分支

2026-09-25。依据B-F已训练15步持续失败，先登记C与tau两因素干预，再执行一个完整科学轮次。

H-BG1八项主判据全部支持：joint在301/302的15步干净／噪声绝对BA下界均>85%，相对baseline配对增益下界均>2pp。最窄绝对下界是301/15/noise的85.0268%。全部质量审计通过，完整1100单元完成，因此按登记规则结案，不追加测试数据。

本结论只限固定干预与已训练15步，不声称全部时长恢复、独立重拟合稳定或新核心推广。次要全矩阵104项下界>85%、4项上界<85%、2项未决；301/304的6步仍失败，304/6有明显退化。两种统计家族分别报告，不把次要110项区间替换主8项判据。

依据：artifacts/branch_bg/round1/protocol.md、summary.json、training_audit.json、independent_audit.json、summary_crosscheck.json；完整解释见reports/branch_bg/FINAL_REPORT.md。
