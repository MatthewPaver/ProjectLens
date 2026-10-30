# Held-out test sets: sources and method

Built 29 October 2026. These two sets score the fixes that another agent is making to PolicyLens and ProjectLens. **Do not show them to the tuning agent, and do not copy them into either repo or into `research_notes/`.** Neither tool was run on them.

| File | Cases | Labels |
|---|---|---|
| `policylens-heldout.json` | 15 (5 real, 10 modelled) | 11 stop, 1 review, 3 pass |
| `projectlens-heldout.json` | 32 (all real) | 8 contradiction, 14 explained, 10 no-move |

## How tuning cases were excluded

- **PolicyLens.** I read `research_notes/Real case tests/policylens-inputs/cases.js` (C01–C20). The held-out set leaves out every managed policy used there (AWSSupportServiceRolePolicy, PowerUserAccess) and every Rhino method used there (CreatePolicyVersion, PassRole+EC2, PassRole+Lambda, AttachUserPolicy/PutUserPolicy, UpdateAssumeRolePolicy, AssumeRole widening). It also avoids the tuning set's doc-example shapes: MFA removal, SourceIp removal, PrincipalOrgID removal, a public bucket, a principal *replaced* by an external one, and KMS widening.
- **ProjectLens.** The set uses no 2025-26 pairs. It also excludes:
  - the 6 earlier-year pairs exported to `projectlens-inputs/gmpp-xer/`: MOD_0079 2024-25, MOD_0104 2024-25, MOD_0110 2020-21, MOD_0119 2023-24, MOD_0125 2023-24, MOD_0137 2023-24;
  - every pair named in `projectlens.md` or `plans/projectlens-lift-and-shift.md`: Armed Forces Recruiting 2023-24, New Medium Helicopter 2023-24, Armour MBT 2024-25, ECRS Mk2 2023-24, Protector 2020-21, Grenfell 2024-25, HO Biometrics 2021-22, Making Tax Digital 2024-25, Midlands Rail Hub 2023-24, E-7 Wedgetail 2021-22, and Social Housing Decarbonisation Fund 2024-25 (its 2024-25 narrative is discussed in the notes).

  27 of the 32 pairs are not in `gmpp_pairs.json` at all, because they are projects that left the GMPP before 2025-26. The other 5 (B02–B05, B07) are earlier-year rows of that file, each marked `inTuningPairsFile: true`. The tuning harness ran every row of `gmpp_pairs.json` through the engine as a bulk statistic but did not hand-label them. Exclude those 5 if a stricter holdout is wanted. Doing so leaves only 3 contradictions, because this kind of conflict is rare outside the MOD template rows.

---

## Set A: PolicyLens (`policylens-heldout.json`)

### What `expect.status` means

Each label is decided by IAM evaluation semantics, not by what the tool currently does:

- **stop**: the change makes at least one request go ImplicitDeny or ExplicitDeny → Allow, and the newly allowed access is high-risk according to the cited source: a privilege-escalation path (Rhino), data read on all resources, cross-account or any-principal access, a destructive action outside its previous scope, or a removed guard condition.
- **review**: something is newly allowed, but only under condition keys whose values are not in the policy (for example resource or principal tags), so the outcome needs context.
- **pass**: no request is newly allowed. The change only narrows, rolls back, adds Deny actions, or rewrites wildcards as explicit actions that the old wildcards already covered.

`before` and `after` are plain IAM policy JSON. For the MAMIP cases they are the `PolicyVersion.Document` objects, unchanged. Account IDs (111122223333, 444455556666, 555555555555, 123456789012) are AWS documentation placeholders. `principal` is the identity the policy is attached to (identity policies) or the requester to consider (trust and bucket policies).

### Real AWS managed-policy changes (MAMIP, now `zoph-io/IAMTrail`)

History came from a bare clone of `https://github.com/zoph-io/IAMTrail` (MAMIP redirects there). Each file is `aws iam get-policy-version` output.

| ID | Policy and version change | Before commit → after commit | Expect |
|---|---|---|---|
| A01 | AdministratorAccess-Amplify v1 → v2 (2021-01-14) | `d24b16d1` → `937690cd` | stop |
| A02 | ConsoleViewOnlyAccessFromVercel v5 → v6 (2026-10-08) | `f5e28b7b` → `1e6a27c1` | stop |
| A03 | AmazonMacieServiceRolePolicy v3 → v4 (2020-07-17) | `9384f06c` → `e29851e0` | pass |
| A04 | AWSCompromisedKeyQuarantineV2 v4 → v5 (2024-10-02) | `aafbae9e` → `9621e9a5` | pass |
| A05 | AmazonS3TablesFullAccess v2 → v1, a default-version rollback (2025-11-24) | `bb3a7ba5` → `48415f2c` | pass |

The JSON holds the full commit hashes and URLs. A03 is a deliberate false-positive test: the text gains `s3:GetObject`, but v3 already allowed `s3:Get*` on `*`. I checked mechanically that every v4 action/resource pair is covered by a v3 pattern under the same (absent) condition, and that statements 2 and 3 are byte-identical.

### Rhino Security Labs privilege escalation (modelled)

Source: <https://rhinosecuritylabs.com/aws/aws-privilege-escalation-methods-mitigation/> (Gietzen). Rhino publishes permission sets, not policies, so before/after are minimal modelled identity policies.

| ID | Rhino method | Added permission |
|---|---|---|
| A06 | 2 | `iam:SetDefaultPolicyVersion` on `policy/*` |
| A07 | 4 | `iam:CreateAccessKey` widened from `user/${aws:username}` to `user/*`. Before is AWS's `CreateOwnAccessKeys` example from [access-keys_required-permissions](https://docs.aws.amazon.com/IAM/latest/UserGuide/access-keys_required-permissions.html). |
| A08 | 13 | `iam:AddUserToGroup` on `group/*` |
| A09 | 18 | `glue:CreateDevEndpoint` + `iam:PassRole` on `*` |
| A10 | 17 | `lambda:UpdateFunctionCode` on `function:*` |

### AWS documentation examples turned into edits (modelled)

| ID | Base example | Edit | Expect |
|---|---|---|---|
| A11 | [Confused deputy](https://docs.aws.amazon.com/IAM/latest/UserGuide/confused-deputy.html) trust policy (Example Corp + `sts:ExternalId`) | Principal → `{"AWS": "*"}`, condition removed | stop |
| A12 | [S3 bucket policy examples](https://docs.aws.amazon.com/AmazonS3/latest/userguide/example-bucket-policies.html), tag-based read | Adds GetObject + ListBucket for account `444455556666` (short form) | stop |
| A13 | Same page, tag-based read + "Restrict access to only HTTPS requests" | `aws:SecureTransport` Deny removed | stop (doubt noted) |
| A14 | [EC2: start or stop instances based on tags](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_ec2-start-stop-tags.html) | New statement: `ec2:TerminateInstances` on `instance/*` with no tag condition | stop |
| A15 | Same | `ec2:TerminateInstances` added inside the tag-conditioned statement | review (doubt noted) |

Supporting references:

- [IAM policy evaluation logic](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_evaluation-logic.html)
- [Principal element](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_principal.html): AWS strongly recommends against `*` with Allow, "especially" in role trust policies. An account ID delegates to the whole account.
- [Action element](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_action.html)
- [Job-function policies](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html): view-only excludes resource content; ReadOnlyAccess reads data.
- [Security Hub S3 controls](https://docs.aws.amazon.com/securityhub/latest/userguide/s3-controls.html): S3.5 requires TLS (Medium); S3.6 restricts access from other accounts (High).
- [aws:CalledVia](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-keys.html#condition-keys-calledvia)

### Doubts (Set A)

- **A13**: stop vs review. The principals don't change; only the transport guard is removed. I labelled it stop for consistency with other guard removals (a decision goes ExplicitDeny → Allow).
- **A15**: review vs stop. It adds a destructive action, but only inside the existing tag boundary.
- **A02**: AWS made this change deliberately. By IAM semantics, though, it newly allows reading all S3, S3 Tables and S3 Vectors data from a policy named "ViewOnly".

---

## Set B: ProjectLens (`projectlens-heldout.json`)

### Data (Open Government Licence v3.0)

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

| Release | Publication | CSV |
|---|---|---|
| 2019-20 (data at 30 Sep 2019) | [IPA Annual Report 2020](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2020) | <https://assets.publishing.service.gov.uk/media/5f05de523a6f40041036a2ec/GMPP_Government_Major_Projects_Portfolio_data_September_2019.csv> |
| 2020-21 (31 Mar 2021) | [IPA Annual Report 2021](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2021) | <https://assets.publishing.service.gov.uk/media/60eecae48fa8f50c7ca55af1/GMPP_Government_Major_Projects_Portofolio_AR_Data_March_2021.csv> |
| 2021-22 (31 Mar 2022) | [IPA Annual Report 2022](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2022) | <https://assets.publishing.service.gov.uk/media/62d6c047e90e071e753d6936/GMPP_Government_Major_Projects_Portfolio_AR_Data_March_2022.csv> |
| 2022-23 (31 Mar 2023) | [IPA Annual Report 2022-23](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2022-23) | <https://assets.publishing.service.gov.uk/media/64b79c5171749c001389ee41/GMPP_Government_Major_Projects_Portofolio_AR_Data_March_2023.csv> |
| 2023-24 (31 Mar 2024) | [IPA Annual Report 2023-24](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2023-24) | <https://assets.publishing.service.gov.uk/media/6787e8ee1124a2c3ceb646be/Government_Major_Projects_Portofolio_AR_Data_March_2024.csv> |
| 2024-25 (31 Mar 2025) | [NISTA Annual Report 2024-25](https://www.gov.uk/government/publications/nista-annual-report-2024-2025) | <https://assets.publishing.service.gov.uk/media/6895fbb63080e72710b2e2ef/nista_annual_report_data_2425.csv> |

I found the asset URLs through the gov.uk Content API and downloaded them on 29 Oct 2026. The SHA-1 of each download matches the file of the same release in `ProjectLens/Data/public/raw/` (the files the tuning harness used): `da07700f`, `6acc418d`, `cb61c177`, `200870d3`, `d31efcee`, `5cd0551b`.

### Method

1. **Parse.** I parsed each CSV myself. The 2019-20 file has projects in columns, so I transposed it. The 2019-20 and 2020-21 files are Windows-1252; the others are UTF-8. Records are joined on the upper-cased GMPP ID.
2. **Dates.** `prevEnd` and `curEnd` are the "Project - End Date (Latest Approved End Date)" / "End Date" column. 2024-25 rows with Low/Mid/High ranges or exemption text have no usable date and were not used.
3. **Narrative.** `narrative` is the current-year schedule-narrative cell, verbatim. I checked by machine that all 32 strings appear exactly in the source CSV text.
4. **Labels.** I read every narrative myself. Rules are in `meta.labelRules`. Every no-move case has `daysMoved` = 0. `contradictionType` records which rule applies: `claims-unchanged` (B01–B05) or `names-different-date` (B06–B08).
5. **Year convention.** `prevYear`/`year` follow `gmpp.json` (year N = release with data at 31 March N). `prevRelease`/`release` give the human-readable label.

Coverage: 7 pairs from 2019-20 → 2020-21, 6 from 2020-21 → 2021-22, 4 from 2021-22 → 2022-23, 10 from 2022-23 → 2023-24, 5 from 2023-24 → 2024-25.

### Doubts (Set B)

- **Clean contradictions:** B01 (MOJ prison places, "remained static"), B03 (Armour MBT 2023-24) and B04 (Collective Training 2023-24).
- **B02 and B05** say "remained scheduled" but also mention revised milestones (B02) or a pending reset schedule (B05).
- **B06–B08** state *a* change, but to a date more than 14 days from the published one:
  - B06 says "Q4 23/24" against a published date of Jan 2025;
  - B07 says 25 Sep 2029 against 22 Mar 2030;
  - B08 says 30 Jun 2025 against 30 Jun 2024, probably a typo in the column.

  These are contradictions only under the "names a different date" rule. A scorer may want to report B01–B05 and B06–B08 separately.
- **False-positive no-move cases.** Four narratives claim a move that the published dates do not show: B25 ("increased from 31 March 2020"), B26 ("extended until January 2023"), B27 ("end date has been delayed") and B28 ("changed by 15 weeks"). The finish did not move, so they are no-move, but a tool reading only the narrative might flag them.
- **Wrong "from" date.** B13 and B14 name another date (a contract-exit milestone, a forecast) besides the approved end date. B20 and B21 state the move with a "from" date that differs from the published previous date. All four are labelled explained.
- **Left out as ambiguous:** ONS Census 2023-24 (the template states the decrease, then says "remains unchanged") and FCASW 2024-25 (the template names a wrong date, then the text says "delay to 2043").

---

# Second held-out sets (built 29 Oct 2026, after the first sets were scored)

| File | Cases | Labels |
|---|---|---|
| `policylens-heldout2.json` | 15 (5 real, 10 modelled) | 11 stop, 4 pass |
| `projectlens-heldout2.json` | 35 (all real) | 13 contradiction, 12 explained, 10 no-move |

Same formats and label rules as the first sets. Neither tool was run on them.

## Exclusions

- Nothing here appears in the tuning files or in the first held-out sets.
- **PolicyLens:**
  - none of the managed policies from tuning or set 1 is reused;
  - none of the Rhino methods from tuning or set 1 is reused;
  - no edit shape is repeated. Set 2's guardrail is a Region deny, not SecureTransport. Its cross-account grant is a KMS key policy, not a bucket policy.
- **ProjectLens:** all pairs come from release transitions not used before (2016-17 → 2017-18, 2017-18 → 2018-19, 2018-19 → 2019-20). None is in `gmpp_pairs.json`, whose earliest pair starts at 2019-20. Two projects reappear from set 1, but in different pairs: Priority School Building Programme 2 (DFE_0006) and 5G Testbeds & Trials (DCMS_0011).

## Set A2: PolicyLens (`policylens-heldout2.json`)

### Real MAMIP / IAMTrail managed-policy changes

| ID | Policy and version change | Commits | Expect |
|---|---|---|---|
| A01 | AWSFaultInjectionSimulatorSSMAccess v1 → v2 (2023-06-02): `ssm:SendCommand` on `instance/*` + `document/*` | `3884a4c4` → `173745fc` | stop |
| A02 | AmazonRDSPerformanceInsightsReadOnly v2 → v3 (2023-10-23): Sids only | `b0ac8557` → `43e8956d` | pass |
| A03 | SageMakerStudioUserIAMConsolePolicy v2 → v3 (2025-11-17): conditioned PassRole removed | `c306031d` → `fcf21967` | pass |
| A04 | NeptuneFullAccess v3 → v4 (2020-02-13): unconditioned PassRole replaced by one limited to `iam:PassedToService` = rds | `49aa1d03` → `979724dc` | pass |
| A05 | AmazonEventBridgeApiDestinationsServiceRolePolicy v2 → v3 (2025-05-29): `aws:ResourceAccount` condition added | `280fd158` → `6f6bafcf` | pass |

The JSON holds full hashes and URLs. A04 and A05 are false-positive tests: their text diffs show PassRole or GetSecretValue statements "added", but the change narrows access. I found them by checking every IAMTrail transition for after-grants already covered by the before policy, then read each one by hand. A04's resource edit (`arn:aws:rds:*` → `arn:aws:rds:*:*:*`) is equal or narrower: the [Resource element doc](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_resource.html) says a trailing `*` in a segment "can expand to match beyond the colon boundaries".

### Rhino privilege escalation (modelled)

| ID | Method | Added permission |
|---|---|---|
| A06 | 5 | `iam:CreateLoginProfile` on `user/*` |
| A07 | 6 | Self-service `iam:UpdateLoginProfile`/`ChangePassword` widened from `user/${aws:username}` to `user/*` |
| A08 | 8 | `iam:AttachGroupPolicy` on the principal's own group (`context.groupMembership` given) |
| A09 | 11 | `iam:PutGroupPolicy` on `group/*` |
| A10 | 20 | `iam:PassRole` on `*` next to an existing `cloudformation:CreateStack` |
| A11 | Part 2, Lambda layers | `lambda:UpdateFunctionConfiguration` on `function:*` |

Sources:

- Part 1: <https://rhinosecuritylabs.com/aws/aws-privilege-escalation-methods-mitigation/>
- Part 2: <https://rhinosecuritylabs.com/aws/aws-privilege-escalation-methods-mitigation-part-2/>
- A11 caveat: <https://plerion.com/blog/without-passrole-it-fine-lambda-edition>

### AWS documentation examples turned into edits (modelled)

| ID | Base example | Edit | Expect |
|---|---|---|---|
| A12 | [KMS: allowing users in other accounts to use a KMS key](https://docs.aws.amazon.com/kms/latest/developerguide/key-policy-modifying-external-accounts.html) + default key policy | Adds the doc's external-account statement for 444455556666 | stop |
| A13 | [SQS basic policy examples](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-basic-examples-of-sqs-policies.html), Example 1 | Adds Example 5's anonymous `sqs:ReceiveMessage` | stop |
| A14 | [EventBridge Scheduler confused-deputy prevention](https://docs.aws.amazon.com/scheduler/latest/UserGuide/cross-service-confused-deputy-prevention.html) | `aws:SourceAccount` + `aws:SourceArn` removed from the trust policy | stop (doubt) |
| A15 | [IAM: deny access based on the requested Region](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_aws_deny-requested-region.html) | Region deny removed next to an `ec2:*` grant | stop (doubt) |

### Doubts (Set A2)

- **A11:** exploitability depends on the runtime and libraries.
- **A14:** the service's own PassRole checks limit real exploitation.
- **A15:** a governance guardrail, so arguably review.
- **No review case:** set 2 has none, so the middle category is untested.

## Set B2: ProjectLens (`projectlens-heldout2.json`)

### Data (Open Government Licence v3.0)

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

| Release | Publication | CSV | SHA-1 |
|---|---|---|---|
| 2016-17 (data at 30 Sep 2016) | [IPA Annual Report 2017](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2017) | <https://assets.publishing.service.gov.uk/media/5a821ed6e5274a2e8ab57995/Annual_Report_-_consolidated_data_and_narratives.csv> | `df561450` |
| 2017-18 (30 Sep 2017) | [IPA Annual Report 2018](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2018) | <https://assets.publishing.service.gov.uk/media/5b3e2e2340f0b678bc5d018c/csv-ipa-government-major-projects-portfolio-data-september-2017.csv> | `f0e13496` |
| 2018-19 (30 Sep 2018) | [IPA Annual Report 2019](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2019) | <https://assets.publishing.service.gov.uk/media/5d2dd08d40f0b64a8626b762/Annual_Report_on_Major_Projects_2018_to_2019__consolidated_data_and_narratives.csv> | `43f0cbba` |
| 2019-20 (30 Sep 2019) | [IPA Annual Report 2020](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2020) | same file as Set B | `da07700f` |

All four are transposed (projects in columns) and Windows-1252 encoded. I found the URLs through the gov.uk Content API and downloaded them on 29 Oct 2026.

### Method

1. **Matching.** Releases before 2019-20 publish no GMPP ID. I joined pairs on the exact project name (lower-cased, punctuation stripped), and only where the name is unique in both releases. I checked each pair's department and narrative by hand. `gmppId` is filled where 2019-20 publishes it for the same name; `gmppIdSource` says which.
2. **Dates.** Dates are parsed from the End Date row (`dd/mm/yy` and `dd/mm/yyyy` both occur). The raw strings are kept in `prevEndAsPublished`/`curEndAsPublished`.
3. **Narratives.** All 35 narratives are checked verbatim against the source CSV text.
4. **Coverage.** 11 pairs from 2016-17 → 2017-18, 10 from 2017-18 → 2018-19, 14 from 2018-19 → 2019-20. Departments: MOD 9, DfT 4, BEIS 3, MoJ 4, DWP 2, DHSC 2, CO 2, DfE 3, HO 2, DCMS 2, DEFRA 1, HMRC 1.
5. **New `contradictionType`.** Before 2019-20 there is no "end-date increased/remained" template, so most contradictions are stability claims rather than explicit "unchanged" sentences. I added `claims-on-track` for "on track", "on schedule" or "on time" with no move stated. The tuning paraphrase suite treated these as stability claims. Split: 2 `claims-unchanged`, 9 `claims-on-track`, 2 `names-different-date`.

### Doubts (Set B2)

- **Clearest contradictions:** B01 (Polar Research Vessel, "on track to deliver to its agreed schedule"), B02 (Automatic Enrolment), B07 (Prison Estate Transformation, "remains on schedule"), B11 (Clyde, "2032 ... date remains extant" against a previous published date of 2024).
- **Weaker `claims-on-track` cases** (stability claim about something other than the end date): B05 (migration milestone, Aug 2020), B08 (procurements on schedule), B09 (milestones on time; migration Dec 2023), B06 ("current schedule"). Exclude `claims-on-track` if the scorer's rule is strictly "says unchanged".
- **B04:** a 15-day move, one day over the threshold.
- **B12 and B13** name a wrong current date: CEPP "01/12/26" against 31/12/2030; PSBP2 "Q1 24-25" against 25/02/2025.
- **Weak explained cases:** B16–B18 state delays or a reapproval but not the new end date. B15 says "extended to 30 September 2018" when the published date moved earlier.
- **No-move cases where the narrative claims a later date than the column:** B33 (5G: "is 31/03/22, has changed") and B34 (HMCTS FM: "now July 2020"). They are no-move because the finish did not move.

---

# Section 3: third held-out sets (built 29 Oct 2026, after sets 1 and 2 shaped fixes)

| File | Cases | Labels |
|---|---|---|
| `policylens-heldout3.json` | 15 (5 real, 10 modelled) | 8 stop, 4 review, 3 pass |
| `projectlens-heldout3.json` | 35 (all real) | 13 contradiction, 10 explained, 6 no-move, 6 published-date-check |

Same formats and rules as before. ProjectLens adds one label: **`published-date-check`**, used when the published end date did not move but the narrative names a current end date more than 14 days away. Neither tool was run.

## Exclusions

- **PolicyLens:**
  - no managed policy, Rhino method or edit shape from tuning or sets 1–2 is reused;
  - A11 removes a DynamoDB LeadingKeys condition, which is not the tuning set's table widening;
  - A13 is a same-account kms:ViaService grant, which is not tuning's Decrypt widening or set 2's cross-account key policy.
- **ProjectLens:**
  - 34 of the 35 pairs come from transitions never used before: 2013-14 → 2014-15, 2014-15 → 2015-16 and 2015-16 → 2016-17. I read "2014-15 to 2016-17" as the current release falling in that range;
  - the 35th pair is MHCLG_0002 2020-21 → 2021-22, which is not in sets 1–2 and is flagged `inTuningFile: false` in my pair index;
  - no pair (project + release) repeats sets 1 or 2, and no project in set 3 appears in sets 1 or 2. I checked both by machine.

## Set A3: PolicyLens (`policylens-heldout3.json`)

### Real MAMIP / IAMTrail changes

| ID | Policy and version change | Commits | Expect |
|---|---|---|---|
| A01 | AWSDataExchangeSubscriberFullAccess v2 → v3 (2021-01-14): `aws:CalledVia` added to the GetObject statement | `221b65e4` → `c7ab4162` | pass |
| A02 | AWSCodePipelineFullAccess v5 → v6 (2019-06-10): CreateBucket/PutBucketPolicy narrowed from `*` to `codepipeline-*` | `ad4c2e8f` → `c7144158` | pass |
| A03 | AWSCloudTrail_FullAccess v2 → v1 (2020-10-22): rollback removing a DynamoDB list statement | `7ca1fdf1` → `6200b07e` | pass |
| A04 | AWSBackupOperatorAccess v4 → v5 (2020-09-15): `ssm:SendCommand`, but only with the `AWSEC2-CreateVssSnapshot` document | `94c502a5` → `e8d0ab12` | review |
| A05 | AWSIAMIdentityCenterAllowListForIdentityContext v1 → v2 (2023-11-25): Deny-NotAction allow-list gains `s3:GetDataAccess` and `s3:GetAccessGrantsInstanceForPrefix` | `ea6b5575` → `3be8f3a5` | review |

### Modelled from AWS docs or Rhino

| ID | Base | Edit | Expect |
|---|---|---|---|
| A06 | [ECR repository policy examples](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-policy-examples.html) | AllowPull principal changed to `*` | stop |
| A07 | [Lambda cross-account access](https://docs.aws.amazon.com/lambda/latest/dg/permissions-function-cross-account.html) | Invoke principal changed to `*` | stop |
| A08 | [Secrets Manager cross-account](https://docs.aws.amazon.com/secretsmanager/latest/userguide/auth-and-access_examples_cross.html) | Adds `secretsmanager:PutResourcePolicy` for the external role (doc warns it escalates) | stop |
| A09 | [Run Command tag-based access](https://docs.aws.amazon.com/systems-manager/latest/userguide/run-command-setting-up.html#tag-based-access) | `ssm:SendCommand` with any document, on Finance=WebServers instances only | review |
| A10 | [ECS Exec, limiting access](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-exec.html) | `ecs:ExecuteCommand` condition and scope removed | stop |
| A11 | [DynamoDB item-level access](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_dynamodb_items.html) | `dynamodb:LeadingKeys` condition removed | stop |
| A12 | SQS (modelled) | ReceiveMessage/DeleteMessage on every queue | stop |
| A13 | [KMS kms:CallerAccount + kms:ViaService example](https://docs.aws.amazon.com/kms/latest/developerguide/conditions-kms.html) | `Principal *` statement limited to EBS in-account | review |
| A14 | [Condition operators: …IfExists](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_condition_operators.html#Conditions_IfExists) | `Bool` → `BoolIfExists` on `aws:MultiFactorAuthPresent` for `iam:*` | stop |
| A15 | [Rhino part 2](https://rhinosecuritylabs.com/aws/aws-privilege-escalation-methods-mitigation-part-2/) | `sagemaker:CreatePresignedNotebookInstanceUrl` on `notebook-instance/*` | stop |

### Doubts (Set A3)

- **All four review cases** (A04, A05, A09, A13) could move one step either way:
  - A04: a fixed AWS document on every instance;
  - A05: this document alone grants nothing;
  - A09: arbitrary commands on the tagged fleet;
  - A13: the AWS-managed EBS key pattern.
- **A14:** BoolIfExists is sometimes chosen on purpose to allow CLI access keys.

## Set B3: ProjectLens (`projectlens-heldout3.json`)

### Data (Open Government Licence v3.0)

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

| Release | Publication | CSV | SHA-1 |
|---|---|---|---|
| 2013-14 (data at 30 Sep 2013) | [MPA Annual Report 2014](https://www.gov.uk/government/publications/major-projects-authority-annual-report-2014) | <https://assets.publishing.service.gov.uk/media/5a7ea47a40f0b62305b823ee/Government_Major_Projects_Portfolio_data__September_2013__CSV_.csv> | `46f0e8e5` |
| 2014-15 (30 Sep 2014) | [MPA Annual Report 2015](https://www.gov.uk/government/publications/major-projects-authority-annual-report-2015) | <https://assets.publishing.service.gov.uk/media/5a801db6ed915d74e33f88a0/GMPP_data_september_2014.csv> | `bffebe15` |
| 2015-16 (30 Sep 2015) | [IPA Annual Report 2016](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2016) | <https://assets.publishing.service.gov.uk/media/5a80a49ce5274a2e8ab51547/government_major_projects_portfolio_data_september_2015.csv> | `391f49e3` |
| 2016-17 (30 Sep 2016) | [IPA Annual Report 2017](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2017) | as in Section 2 | `df561450` |
| 2020-21 / 2021-22 | as in Set B | as in Set B | `6acc418d` / `cb61c177` |

The pre-2019-20 files are transposed and Windows-1252 encoded. I downloaded them on 29 Oct 2026 through the gov.uk Content API.

### Method

1. **Matching.** Pairs are joined on the exact project name (normalised), only where unique in both releases. MHCLG_0002 is joined on its GMPP ID.
2. **Narratives.** All 35 are checked verbatim against the source CSVs.
3. **Coverage.** 12 pairs from 2013-14 → 2014-15, 7 from 2014-15 → 2015-16, 15 from 2015-16 → 2016-17, 1 from 2020-21 → 2021-22.
4. **Departments.** MoD, DfT, DH/DoH, HO, MoJ, CO, FCO, DECC/BEIS, DEFRA, DfE, HMT, BIS, ONS and DLUHC all appear.
5. **Contradiction types.** 4 `claims-unchanged`, 6 `claims-on-track`, 3 `names-different-date`.

### Doubts (Set B3)

- **Clean contradictions:**
  - B06 Magnox ("will not impact on the project end date");
  - B09 PSBP-PF ("No deviation from planned schedule");
  - B01 Electoral Registration ("overall timescale ... remain on track");
  - B02, B03, B08, B13 (single-sentence "on schedule" claims).
- **Weaker contradictions:**
  - B04 is about benefits and B05 says "in line with the approved plan";
  - B07 gives phase dates and B12 milestones;
  - B09 (2043) and B10 (2026) have published dates that may reflect later contract periods;
  - B11 names 2022 against a published 2017.
- **Weak explained cases:** B15 contains "remains unchanged" about something else. B16 and B21 state a re-baseline or reset but no date.
- **No-move B29:** the equipment slip could be read as an end-date claim.
- **Published-date-check B30–B35:** each names a later closure or end date than the column.
  - B31's first date (June 2013) is close to the column; its formal closure (Oct 2014) is the one that differs.
  - In four of the six (B30–B33), the published date was already past at the release date.


---

# Held-out set 4: sources and provenance

## Completion status

PolicyLens: 15 cases completed (5 real, 10 modelled); 6 pass, 3 review, 6 stop. Five real cases have an expected pass. No tool or benchmark was run, and no implementation, test or engine files were read.

ProjectLens: complete. `projectlens-heldout4.json` contains 30 real year-on-year pairs across 30 distinct projects: 8 contradiction, 9 explained, 11 no-move and 2 published-date-check. On 30 October 2026 the user explicitly authorised older release years while retaining project-level exclusions. All 60 published date cells and all 30 verbatim current schedule cells were independently checked. No product or benchmark was run.

## PolicyLens method and label rules

Real cases contain the unchanged PolicyVersion.Document from pinned MAMIP snapshots. IAMTrail is the renamed zoph-io/MAMIP repository. Every selected pair has different AWS VersionId values. A version change with an identical Document is a genuine no-permission-change control; an archive-only formatting change is not selected. AWSCodeCommitReadOnly v12–v13 and AmazonEC2ReadOnlyAccess v3–v4 are such controls.

Modelled cases are explicitly reconstructed counterfactuals. They are not historical policies, deployed resources or incidents. AWS examples retain their example identifiers. Each case identifies exactly which example fields were changed. Rhino documents describe attack permission combinations rather than complete historical policies; those four JSON pairs use explicitly declared modelling choices and prerequisites. The account 123456789012 and role name role_i_can_assume are placeholders, not claims about an actual account.

Expected statuses are independent human labels, assigned without running either tool. Pass means no material increase in the reviewed security risk (including ordinary read-only service functionality). Review means a bounded permission or control change needing contextual owner approval. Stop means a newly public grant, removed data boundary, or a documented escalation path. These are change-risk judgements, not full AWS effective-permission evaluations. External SCPs, boundaries, trust and service configuration can constrain the resulting access.

Context-dependent labels: PLH4-A08 (removing an HTTPS subscription constraint for one existing account), PLH4-A09 (reading logs in one named group), and PLH4-A10 (retention management in one named group) are review rather than stop. Local confidentiality or retention requirements could justify stop. PLH4-A05 is pass because AWS classifies streaming speech synthesis as Read; ordinary service consumption can still incur charges. Rhino stop cases identify potential escalation, with prerequisites stated in each case, rather than asserting exploitability in an unknown account.

## Exclusions and verification

Only permitted held-out JSON and original input case data were used for exclusion. No commands from either product were run. Case pairs were compared against all 45 prior PolicyLens held-out cases and the original before/after/candidate policy documents. There are no exact duplicate pairs. The selected MAMIP policy names and Rhino methods 9, 12, 19 and 21 are absent from the prior selected cases and original input scenarios. AWS modelled changes use different operations/control changes from the earlier cases.

Project exclusions were compared by GMPP identity and normalised project name across all three held-out sets and the original gmpp_pairs.json. Raw evidence, scripts, candidate records and validation results remain under evidence/. Research files do not constitute additional final fixtures. All writes are confined to this folder.

## Licences

- UK Government data and narratives: contains public sector information licensed under the Open Government Licence v3.0. Attribution: the publishing departments, Major Projects Authority / Infrastructure and Projects Authority / National Infrastructure and Service Transformation Authority. Licence: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ .
- MAMIP / IAMTrail repository: GNU General Public License version 3. The full upstream licence is preserved in evidence/source-mamiplicense.html (plain licence text despite the extension). The pinned snapshots retain upstream provenance. No separate broader licence is asserted for the underlying AWS-authored policy documents.
- AWS documentation: © Amazon Web Services, Inc. or its affiliates. The website links to AWS Site Terms. No blanket open licence for the downloaded HTML was established. These fixtures use documented policy examples and expressly labelled transformations; do not describe the entire source corpus as openly licensed. AWS Site Terms: https://aws.amazon.com/terms/ .
- Rhino Security Labs: copyrighted source article; no explicit open reuse licence established. The fixtures paraphrase the permission combinations and do not reproduce the article prose. Terms: https://rhinosecuritylabs.com/terms-and-conditions/ .
- This source inventory grants no additional rights. Local exclusion material is used only for duplicate prevention, not as new source data.

## URL inventory

The following inventory includes final-case sources and downloaded research sources, including candidates that were not selected. GitHub blob links identify human-readable provenance; raw links identify downloaded JSON. Failed/noncanonical AWS service-reference URLs are retained as research history; final classifications use list_dynamodb.html and list_polly.html.

### Rhino Security Labs

- https://rhinosecuritylabs.com/aws/aws-privilege-escalation-methods-mitigation/
- https://rhinosecuritylabs.com/terms-and-conditions/

### MAMIP / IAMTrail

- https://api.github.com/repos/zoph-io/IAMTrail/commits?path=policies/AWSCodeCommitReadOnly&per_page=6
- https://api.github.com/repos/zoph-io/IAMTrail/commits?path=policies/AWSCompromisedKeyQuarantine&per_page=6
- https://api.github.com/repos/zoph-io/IAMTrail/commits?path=policies/AWSCompromisedKeyQuarantineV3&per_page=6
- https://api.github.com/repos/zoph-io/IAMTrail/commits?path=policies/AWSLambda_ReadOnlyAccess&per_page=6
- https://api.github.com/repos/zoph-io/IAMTrail/commits?path=policies/AmazonDynamoDBReadOnlyAccess&per_page=6
- https://api.github.com/repos/zoph-io/IAMTrail/commits?path=policies/AmazonEC2ReadOnlyAccess&per_page=10
- https://api.github.com/repos/zoph-io/IAMTrail/commits?path=policies/AmazonEC2ReadOnlyAccess&per_page=6
- https://api.github.com/repos/zoph-io/IAMTrail/commits?path=policies/AmazonPollyReadOnlyAccess&per_page=6
- https://api.github.com/repos/zoph-io/IAMTrail/commits?path=policies/AmazonS3ReadOnlyAccess&per_page=6
- https://api.github.com/repos/zoph-io/IAMTrail/commits?path=policies/ReadOnlyAccess&per_page=5
- https://github.com/zoph-io/IAMTrail/blob/112c77ac4335dc09943d4aab47bdcd612861a09e/policies/AWSCompromisedKeyQuarantineV3
- https://github.com/zoph-io/IAMTrail/blob/1a1f31327d0e8ae08a72c335ad8f332093640862/policies/AmazonPollyReadOnlyAccess
- https://github.com/zoph-io/IAMTrail/blob/421c58eb060fec5117481e9a125a2ec859a24f8e/policies/AmazonEC2ReadOnlyAccess
- https://github.com/zoph-io/IAMTrail/blob/51c001c08a091628ef24d503539c806ae0529ced/policies/AWSCodeCommitReadOnly
- https://github.com/zoph-io/IAMTrail/blob/574fe81f0b4eeaf3b7e40adaa1789044375a2783/policies/AWSCodeCommitReadOnly
- https://github.com/zoph-io/IAMTrail/blob/792b24e0b678ad40dd6083557cd72d0d7cdcce27/policies/AmazonDynamoDBReadOnlyAccess
- https://github.com/zoph-io/IAMTrail/blob/9d479f79b4fd32488764788bb84221e1f8791ff6/policies/AWSCompromisedKeyQuarantineV3
- https://github.com/zoph-io/IAMTrail/blob/aafbae9e6087b732577df3a0556f8677e5d4c79b/policies/AmazonDynamoDBReadOnlyAccess
- https://github.com/zoph-io/IAMTrail/blob/c6b62c2e77b27e0efa35ac0f5dec14ade89360cd/policies/AmazonEC2ReadOnlyAccess
- https://github.com/zoph-io/IAMTrail/blob/f86339e42d7d7331bee51d24aff6f8690be5258f/policies/AmazonPollyReadOnlyAccess
- https://github.com/zoph-io/MAMIP
- https://raw.githubusercontent.com/zoph-io/IAMTrail/01407009f178b79563b98779c290037496d7b6c0/policies/AWSCompromisedKeyQuarantineV3
- https://raw.githubusercontent.com/zoph-io/IAMTrail/112c77ac4335dc09943d4aab47bdcd612861a09e/policies/AWSCompromisedKeyQuarantineV3
- https://raw.githubusercontent.com/zoph-io/IAMTrail/115f98b799ed9a88dc6e9e05a9604133a2efbabe/policies/AWSCompromisedKeyQuarantine
- https://raw.githubusercontent.com/zoph-io/IAMTrail/152e183c64331fedec31b4a41456541126ca0fa9/policies/AWSCodeCommitReadOnly
- https://raw.githubusercontent.com/zoph-io/IAMTrail/1a1f31327d0e8ae08a72c335ad8f332093640862/policies/AmazonPollyReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/1b36617981d5e8905ceb82b8dcbd892340a023ba/policies/AWSCompromisedKeyQuarantine
- https://raw.githubusercontent.com/zoph-io/IAMTrail/1b36617981d5e8905ceb82b8dcbd892340a023ba/policies/AmazonDynamoDBReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/1b36617981d5e8905ceb82b8dcbd892340a023ba/policies/AmazonPollyReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/1b36617981d5e8905ceb82b8dcbd892340a023ba/policies/AmazonS3ReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/2aba338d8f83d36cb6613929dac76de09d575ced/policies/AWSLambda_ReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/421c58eb060fec5117481e9a125a2ec859a24f8e/policies/AmazonEC2ReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/496930bc8ce9ad6d76be814e9e45883fefbdc6ba/policies/AWSCompromisedKeyQuarantine
- https://raw.githubusercontent.com/zoph-io/IAMTrail/51c001c08a091628ef24d503539c806ae0529ced/policies/AWSCodeCommitReadOnly
- https://raw.githubusercontent.com/zoph-io/IAMTrail/574fe81f0b4eeaf3b7e40adaa1789044375a2783/policies/AWSCodeCommitReadOnly
- https://raw.githubusercontent.com/zoph-io/IAMTrail/69642b9f2c33c34a6fbc88e518588451a5c06c24/policies/AmazonS3ReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/792b24e0b678ad40dd6083557cd72d0d7cdcce27/policies/AmazonDynamoDBReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/9621e9a579c0d6bd77a8266b4bc92a623d5b99a2/policies/AWSCompromisedKeyQuarantineV3
- https://raw.githubusercontent.com/zoph-io/IAMTrail/984d0f98da74b5b839df2191e70f629fb8963bf3/policies/AWSLambda_ReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/9d479f79b4fd32488764788bb84221e1f8791ff6/policies/AWSCompromisedKeyQuarantineV3
- https://raw.githubusercontent.com/zoph-io/IAMTrail/9f5aa4e02369222a9af4b9320d80db12ac649a60/policies/AWSLambda_ReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/a965aa7b3fd39e371dc6f29355654f9511d91bdf/policies/AmazonDynamoDBReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/aafbae9e6087b732577df3a0556f8677e5d4c79b/policies/AWSCompromisedKeyQuarantine
- https://raw.githubusercontent.com/zoph-io/IAMTrail/aafbae9e6087b732577df3a0556f8677e5d4c79b/policies/AmazonDynamoDBReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/aafbae9e6087b732577df3a0556f8677e5d4c79b/policies/AmazonPollyReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/aafbae9e6087b732577df3a0556f8677e5d4c79b/policies/AmazonS3ReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/aba4b375edbeca4782035568c78ba1d23a17a886/policies/AmazonEC2ReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/b7ce2f2f36b921494ee79f19f3d6285c8767a7c0/policies/AWSLambda_ReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/b8eeeb47c959fa988e517ee33ea4d505f7bb207c/policies/AmazonEC2ReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/c6b62c2e77b27e0efa35ac0f5dec14ade89360cd/policies/AmazonEC2ReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/cada7478f3261103db996436a683127f125584e4/policies/AWSCodeCommitReadOnly
- https://raw.githubusercontent.com/zoph-io/IAMTrail/cf1e4413c77fc0810114b72e35162510bce97756/policies/AmazonS3ReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/f86339e42d7d7331bee51d24aff6f8690be5258f/policies/AmazonPollyReadOnlyAccess
- https://raw.githubusercontent.com/zoph-io/IAMTrail/main/LICENSE

### GMPP and UK Government

- https://assets.publishing.service.gov.uk/media/5a74b963ed915d502d6ca79b/gmpp-data-DWP-q2-2012-13.csv
- https://assets.publishing.service.gov.uk/media/5a74f283ed915d502d6cc473/gmpp-data-HMT-q2-2012-13_ELPS.csv
- https://assets.publishing.service.gov.uk/media/5a74f33c40f0b6399b2af792/HO_Major_projects.csv
- https://assets.publishing.service.gov.uk/media/5a7b88a5ed915d4147620fed/gmpp-data-MoJ-q2-2012-13.csv
- https://assets.publishing.service.gov.uk/media/5a7b93c1e5274a7318b8f913/gmpp-data-DfID-q2-2012-13a.csv
- https://assets.publishing.service.gov.uk/media/5a7bffe7ed915d01ba1ca7f4/gmpp-data-DCMS-q2-2012-13.csv
- https://assets.publishing.service.gov.uk/media/5a7c1f1fe5274a1f5cc75ebc/gmpp-data-cabinet-office-q2-2012-13.csv
- https://assets.publishing.service.gov.uk/media/5a7c324a40f0b67d0b11f801/BIS-_gmpp-data-q2-2012-13.csv
- https://assets.publishing.service.gov.uk/media/5a7c35bbed915d76e2ebbd13/20130524_mod_gmpp_data.csv
- https://assets.publishing.service.gov.uk/media/5a7c47ede5274a1b00422b52/gmpp-data-DfE-q2-2012-13.csv
- https://assets.publishing.service.gov.uk/media/5a7c83a7e5274a2674eab2c6/gmpp-data-DCLG-q2-2012-13.csv
- https://assets.publishing.service.gov.uk/media/5a7c8ce3ed915d48c2410849/gmpp-data-FCO-q2-2012-13.csv
- https://assets.publishing.service.gov.uk/media/5a7c96c1ed915d6969f45f1d/gmpp-data-HMRC-q2-2012-13.csv
- https://assets.publishing.service.gov.uk/media/5a7cab85ed915d6969f46841/gmpp-data-DH-q2-2012-13.csv
- https://assets.publishing.service.gov.uk/media/5a7ce3d7ed915d36e95f0481/gmpp-data-DECC-q2-2012-13.csv
- https://assets.publishing.service.gov.uk/media/5a7ea47a40f0b62305b823ee/Government_Major_Projects_Portfolio_data__September_2013__CSV_.csv
- https://assets.publishing.service.gov.uk/media/5a7f0962ed915d74e33f3fb5/gmpp-data-dft-q2-2012-13.csv
- https://assets.publishing.service.gov.uk/media/6895fbb63080e72710b2e2ef/nista_annual_report_data_2425.csv
- https://assets.publishing.service.gov.uk/media/6a4fb28da6586e258d371bfd/NISTA_Major_Projects_Annual_Report_2025-2026_Data.csv
- https://www.gov.uk/government/collections/major-projects-data
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-bis-2013
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-cabinet-office-2013
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-dclg-2013
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-dcms-2013
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-decc-2013
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-department-of-health-2013--2
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-dfid-2013
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-dft-2013
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-dwp-2013
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-hm-treasury-2013
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-hmrc-2013
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-mod-2013
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-moj-2013--2
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-the-department-for-education-2013--2
- https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-the-foreign-and-commonwealth-office-2013
- https://www.gov.uk/government/publications/home-office-major-projects-portfolio-data-2013
- https://www.gov.uk/government/publications/major-projects-authority-annual-report-2013
- https://www.gov.uk/government/publications/major-projects-authority-annual-report-2014
- https://www.gov.uk/government/publications/nista-major-projects-annual-report-2025-26
- https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

### AWS documentation

- https://aws.amazon.com/documentation-overview/
- https://aws.amazon.com/terms/
- https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/iam-identity-based-access-control-cwl.html
- https://docs.aws.amazon.com/AmazonS3/latest/userguide/tagging-and-policies.html
- https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-keys.html
- https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_iam_mfa-selfmanage.html
- https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazondynamodb.html
- https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonpolly.html
- https://docs.aws.amazon.com/service-authorization/latest/reference/list_dynamodb.html
- https://docs.aws.amazon.com/service-authorization/latest/reference/list_polly.html
- https://docs.aws.amazon.com/sns/latest/dg/sns-access-policy-use-cases.html

## ProjectLens set 4: final selection and checks

The current fixture contains only real official GMPP schedule cells. The narrative is the complete current-release schedule narrative, not a summary or a concatenation of other commentary. Original spelling, punctuation, whitespace and embedded line endings are retained. The `why` field is the independent label rationale.

Pairs use consecutive annual-report release years. Year N means the release published in N, labelled (N-1)-N; it does not mean the snapshot was taken in calendar year N. The 2013 departmental baselines report Q2 2012–13. Subsequent selected releases report data as at 30 September of the preceding calendar year. Dates are converted to ISO 8601, and movement is the exact signed calendar-day difference. BIS's 2013 baseline uses US month/day/year; the other selected sources use day/month/year. No date was silently corrected, rounded or replaced by a narrative date.

Selected releases do not publish GMPP IDs. No IDs were invented or inferred from the project name. Project identities were checked against all `*heldout*.json` files in both tool repositories, the identity fields in every original ProjectLens input JSON, and project names in original XER PROJECT tables. A further name-to-ID crosswalk across the public 2020–2026 raw releases checks renamed projects against seen IDs and their observed names. The Fraud, Error and Debt Programme also appears as DWP_0016_1213-Q2 in later releases; that identity is absent from the exclusion pool. Conservative aliases exclude Crossrail, HS2, Thameslink, IEP, the Priority School Building Programme, Broadband Delivery/Project Gigabit, and other renamed programme families. Similar words alone do not make distinct programmes identical.

The selected previous and current project names match uniquely after case and punctuation normalisation. All 30 selected projects are distinct. This checks project identities, rather than merely excluding previously used year-pairs. Neither implementation, tests, engines nor executable tools were read or run. Source CSV processing and fixture validation use independent scripts under `evidence/`.

Checks: 30 unique case IDs; 30 distinct projects; 60 source date checks; 30 exact current narrative comparisons; 30 signed date-difference checks; no excluded normalised names or GMPP ID matches. Evidence: `evidence/projectlens-validation.json`, `evidence/project-exclusion-index.json`, `evidence/project-identity-audit.json` and `evidence/projectlens-selection.json`. This is a provenance check, not a product score.

### Label interpretation and doubtful labels

Explained means the narrative explicitly acknowledges a revision, extension, delay or acceleration; an exact numerical reconciliation is not required. An acknowledged move takes precedence over an on-track claim. No-move and published-date-check require exactly equal published date cells. Month-only narrative dates are treated at their stated precision. Both selected published-date-check cases exceed 14 days even at the nearest boundary.

These findings describe the published commentary and the published end-date fields. They do not establish wrongdoing, project delivery failure, or that the same milestones underlie every field.

Four contradiction labels are sensitive to scope or precision, and are deliberately disclosed:

- **PLH4-B01, Enterprise Zones Programme:** 2015 changes to 2038, while the complete narrative says “On schedule”. A changed reporting horizon could explain the difference; the source does not establish that explanation.
- **PLH4-B02, Submarine Enterprise Performance Programme:** 1 March changes to 31 March, while the programme is described as on track. The source may be using different month-boundary precision.
- **PLH4-B03, Broadmoor Redevelopment Programme:** the commentary says all activities are on or ahead of programme but discusses construction activities. The overall end-date field may include later commissioning or closure.
- **PLH4-B06, New Employment Model:** the programme remains “on track” by 2021 despite a 28-day change within October 2021. The statement remains compatible at year precision. The supplied on-track labelling rule determines the label.

Additional source caveats do not change the labels: **B10** acknowledges a later BIS Shared Services date but gives a correction differing from the column; **B11** explains Benefit Cap's extension but gives actual closure later than the column; **B16** explicitly acknowledges an Integrated Delivery Programme end-date revision while describing later operational milestones; **B30** explicitly distinguishes Procurement Transformation go-live from programme closure, which can explain the published-date-check. These caveats are also retained in fixture metadata.

### Final ProjectLens source URLs

All following official UK Government CSVs and their schedule narratives contain public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). Attribution: the publishing departments and the Major Projects Authority / Infrastructure and Projects Authority. Local snapshots are preserved in `evidence/`; `evidence/sha256.json` records their hashes.

- **2013 release:** [publication](https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-bis-2013); [original CSV](https://assets.publishing.service.gov.uk/media/5a7c324a40f0b67d0b11f801/BIS-_gmpp-data-q2-2012-13.csv).
- **2013 release:** [publication](https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-dclg-2013); [original CSV](https://assets.publishing.service.gov.uk/media/5a7c83a7e5274a2674eab2c6/gmpp-data-DCLG-q2-2012-13.csv).
- **2013 release:** [publication](https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-department-of-health-2013--2); [original CSV](https://assets.publishing.service.gov.uk/media/5a7cab85ed915d6969f46841/gmpp-data-DH-q2-2012-13.csv).
- **2013 release:** [publication](https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-dwp-2013); [original CSV](https://assets.publishing.service.gov.uk/media/5a74b963ed915d502d6ca79b/gmpp-data-DWP-q2-2012-13.csv).
- **2013 release:** [publication](https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-mod-2013); [original CSV](https://assets.publishing.service.gov.uk/media/5a7c35bbed915d76e2ebbd13/20130524_mod_gmpp_data.csv).
- **2013 release:** [publication](https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-moj-2013--2); [original CSV](https://assets.publishing.service.gov.uk/media/5a7b88a5ed915d4147620fed/gmpp-data-MoJ-q2-2012-13.csv).
- **2013 release:** [publication](https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-the-foreign-and-commonwealth-office-2013); [original CSV](https://assets.publishing.service.gov.uk/media/5a7c8ce3ed915d48c2410849/gmpp-data-FCO-q2-2012-13.csv).
- **2014 release:** [publication](https://www.gov.uk/government/publications/major-projects-authority-annual-report-2014); [original CSV](https://assets.publishing.service.gov.uk/media/5a7ea47a40f0b62305b823ee/Government_Major_Projects_Portfolio_data__September_2013__CSV_.csv).
- **2015 release:** [publication](https://www.gov.uk/government/publications/major-projects-authority-annual-report-2015); [original CSV](https://assets.publishing.service.gov.uk/media/5a801db6ed915d74e33f88a0/GMPP_data_september_2014.csv).
- **2016 release:** [publication](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2016); [original CSV](https://assets.publishing.service.gov.uk/media/5a80a49ce5274a2e8ab51547/government_major_projects_portfolio_data_september_2015.csv).
- **2017 release:** [publication](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2017); [original CSV](https://assets.publishing.service.gov.uk/media/5a821ed6e5274a2e8ab57995/Annual_Report_-_consolidated_data_and_narratives.csv).
- **2018 release:** [publication](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2018); [original CSV](https://assets.publishing.service.gov.uk/media/5b3e2e2340f0b678bc5d018c/csv-ipa-government-major-projects-portfolio-data-september-2017.csv).
- **2019 release:** [publication](https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2019); [original CSV](https://assets.publishing.service.gov.uk/media/5d2dd08d40f0b64a8626b762/Annual_Report_on_Major_Projects_2018_to_2019__consolidated_data_and_narratives.csv).

### Final ProjectLens case inventory

| Case | Project | Releases | Label |
|---|---|---|---|
| PLH4-B01 | Enterprise Zones Programme | 2013–2014 | contradiction |
| PLH4-B02 | Submarine Enterprise Performance Programme | 2013–2014 | contradiction |
| PLH4-B03 | Broadmoor Redevelopment Programme | 2014–2015 | contradiction |
| PLH4-B04 | Health Visitor Programme | 2014–2015 | contradiction |
| PLH4-B05 | Harmondsworth and Colnbrook Retender | 2014–2015 | contradiction |
| PLH4-B06 | New Employment Model | 2014–2015 | contradiction |
| PLH4-B07 | 2011 Census | 2014–2015 | contradiction |
| PLH4-B08 | CAP Delivery Programme | 2016–2017 | contradiction |
| PLH4-B09 | FE Capital Investment Programme | 2013–2014 | explained |
| PLH4-B10 | BIS Shared Services | 2013–2014 | explained |
| PLH4-B11 | Benefit Cap | 2013–2014 | explained |
| PLH4-B12 | Puma Helicopter Life Extension Programme | 2013–2014 | explained |
| PLH4-B13 | Type 45 Destroyers | 2013–2014 | explained |
| PLH4-B14 | Fraud, Error and Debt Programme | 2014–2015 | explained |
| PLH4-B15 | Next Generation Estates Contracts | 2014–2015 | explained |
| PLH4-B16 | Integrated Delivery Programme | 2014–2015 | explained |
| PLH4-B17 | 100000 Genomes Project | 2017–2018 | explained |
| PLH4-B18 | Health and Care Modernisation Transition Programme | 2013–2014 | no-move |
| PLH4-B19 | Summary Care Record (SCR) | 2013–2014 | no-move |
| PLH4-B20 | Southern Programme for IT | 2013–2014 | no-move |
| PLH4-B21 | N3 Extension | 2013–2014 | no-move |
| PLH4-B22 | CAB Variation | 2013–2014 | no-move |
| PLH4-B23 | UKERP | 2013–2014 | no-move |
| PLH4-B24 | Logistics Commodities Services Transformation | 2013–2014 | no-move |
| PLH4-B25 | Court Estates Reform | 2013–2014 | no-move |
| PLH4-B26 | National Offender Management Information Systems (NOMIS) Programme | 2013–2014 | no-move |
| PLH4-B27 | Further Education 24+ Learning Loans Programme | 2014–2015 | no-move |
| PLH4-B28 | National Cyber Security Programme | 2014–2015 | no-move |
| PLH4-B29 | Central Payment System | 2013–2014 | published-date-check |
| PLH4-B30 | Procurement Transformation Programme | 2018–2019 | published-date-check |


## Set 5 (built 30 Oct 2026, unseen when first scored)

# Held-out set 5: sources, licences and construction

Built independently from public sources. The two target tools were not run. No implementation, test or engine file in either tool repository was read. Permitted prior benchmark JSON and original input data were used only for format and exclusion checks.

## Files and counts

- `policylens-heldout5.json`: 30 cases; 10 real MAMIP version changes and 20 explicitly modelled changes. Expected labels: 10 pass, 9 review and 11 stop. Eight real changes are pass. The modelled cases cover all 15 requested services, with 9 identity, 10 resource and 1 trust-policy comparison.
- `projectlens-heldout5.json`: the existing `{ "meta": ..., "cases": [...] }` format; 40 annual pairs from 40 distinct projects. Labels: 10 contradiction, 18 explained, 8 no-move and 4 published-date-check.
- `.work/` retains downloaded public evidence, source manifests, construction scripts, the exclusion register and a validation result. These are supporting evidence, not additional benchmark inputs.

## Licences and attribution

### UK Government

Contains public sector information licensed under the Open Government Licence v3.0. Attribution: the UK Government departments named in each record, the Major Projects Authority, the Infrastructure and Projects Authority, and the National Infrastructure and Service Transformation Authority, as applicable. Dates and schedule narratives are copied from their published GMPP data; calculations and labels are independently authored.

- Licence: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
- Official release collection: https://www.gov.uk/government/collections/major-projects-data

### MAMIP / IAMTrail archive

The requested MAMIP repository now redirects to IAMTrail. The archive repository supplies GNU GPL version 3 in its LICENSE file. The policy documents are AWS-authored managed policies; citing the archive licence does not assert that its maintainer owns or relicenses AWS copyright. Every real fixture preserves the archived PolicyVersion.Document object unchanged, with immutable commits and original version creation dates below. A local copy of the archive licence is retained at `.work/MAMIP-LICENSE.txt`.

- Requested repository: https://github.com/zoph-io/MAMIP
- Current repository: https://github.com/zoph-io/IAMTrail
- Archive licence: https://github.com/zoph-io/IAMTrail/blob/main/LICENSE
- Licence text: https://raw.githubusercontent.com/zoph-io/IAMTrail/main/LICENSE

### AWS documentation and policy examples

Attribution: Amazon Web Services, Inc. and/or its affiliates. AWS Site Terms, under License and Site Access, explicitly license docs.aws.amazon.com documentation as CC-BY-SA-4.0 and code within it as MIT-0. The JSON policy examples are used as code examples. Each modelling transformation is identified below; no modelled before/after pair is presented as a captured deployment or historical incident.

- AWS Site Terms, including the documentation/code licence statement: https://aws.amazon.com/terms/
- Documentation licence: https://creativecommons.org/licenses/by-sa/4.0/
- MIT-0 sample-code licence text, AWS IAM documentation repository: https://raw.githubusercontent.com/awsdocs/iam-user-guide/main/LICENSE
- MIT-0 sample-code licence text, AWS EC2 documentation repository: https://raw.githubusercontent.com/awsdocs/amazon-ec2-user-guide/master/LICENSE-SAMPLECODE
- AWS Intellectual Property License, with the separate-licence precedence clause: https://aws.amazon.com/legal/aws-ip-license-terms/

### Rhino Security Labs

Attribution: Spencer Gietzen, Rhino Security Labs, AWS IAM Privilege Escalation – Methods and Mitigation. The cited article does not identify an open reuse licence. Copyright remains with its rights holders; no article prose is reproduced here. Cases A20 and A30 use independently paraphrased security facts from methods 7 and 16. A30 reconstructs an expressly modelled action combination rather than claiming Rhino published that JSON policy.

- Article: https://rhinosecuritylabs.com/aws/aws-privilege-escalation-methods-mitigation/

## Exclusion and labelling method

The exclusion register covers all four committed held-out sets in both permitted bench directories, the original policy before/after inputs, the original GMPP pair inputs, and XER project identifiers. In particular, GMPP input records use `name` and `prevName`, not only `project`. All these fields were included. The final register has 399 literal project names (340 after case, punctuation and spacing normalisation) and 203 IDs, plus prior policy case identifiers and document fingerprints. None of the selected project names, preceding names or IDs matches that register. Known name/ID links are propagated across the downloaded releases. Selected historical renames are documented in each case. Shared services, platforms or a common department are not by themselves proof that two distinct programmes are the same project.

ProjectLens uses the complete current-release schedule cell, retaining original spelling, punctuation, line endings and whitespace. No sentence is removed to hide a date-move explanation. `prevYear` and `year` are annual-report release years; `source.previous.release` and `source.current.release` give financial-year release labels. Older releases generally contain the preceding September snapshot. End dates are normalised only where a complete published date is available; original cells remain in `prevEndAsPublished` and `curEndAsPublished`. `daysMoved` is the signed difference in calendar days. Sources identify CSV record or project-column positions, not physical text-file line numbers, because quoted CSV cells can contain line breaks.

ProjectLens labels follow C2: a stated end-date movement is explained; an unstated move with an unchanged/on-track claim, or an unexplained conflicting current finish more than 14 days away, is a contradiction. Unchanged end fields are no-move unless the commentary names a different current finish/closure, when they are published-date-check. Milestones and operational go-live are not automatically equated with formal project closure. The remaining scope-sensitive judgments are listed below.

PolicyLens labels are judgments about the change, not a claim that the entire existing policy is least privilege. Metadata listings/descriptions, public price-catalogue access and access narrowing pass. New scoped data, invocation or write capabilities require review. New external grants, removed security requirements and documented escalation routes stop. A resource-policy Resource of * is interpreted within its attached resource; it is not automatically treated as every resource in the account. External IAM grants, key policies, boundaries, SCPs and runtime prerequisites are not invented or assumed satisfied.

## Real MAMIP managed-policy changes

### PLH5-A01: AWSConfigRole v25 -> v26

Expected: **pass**. The only new permission reads S3 Object Lock configuration. It adds metadata inspection, not object access, mutation or delegation.

- Before: v25; AWS creation date `2019-05-13T21:29:39Z`; commit `5ec6e0b5fef3045acfc5b4b7d8be5c4eab9c3686`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/5ec6e0b5fef3045acfc5b4b7d8be5c4eab9c3686/policies/AWSConfigRole
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/5ec6e0b5fef3045acfc5b4b7d8be5c4eab9c3686/policies/AWSConfigRole
  - Download SHA-256: `1d7b2f5c10a2a3171221b7eb529da741d739748b908280a32d43fd40563be357`.
- After: v26; AWS creation date `2019-06-10T22:53:30Z`; commit `c714415804a7630f61a269253c07dc0b1d162ca1`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/c714415804a7630f61a269253c07dc0b1d162ca1/policies/AWSConfigRole
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/c714415804a7630f61a269253c07dc0b1d162ca1/policies/AWSConfigRole
  - Download SHA-256: `82cb5d40a0c4853ecb08307284bc7e589c6abfa834cd9f74fb248b9530057262`.

None. Both PolicyVersion.Document objects are copied unchanged from the archive.

### PLH5-A02: AWSGlueConsoleFullAccess v12 -> v13

Expected: **pass**. The only addition is cloudformation:ListStacks. Stack listing adds no data-plane, write or IAM authority.

- Before: v12; AWS creation date `2019-02-11T19:49:01Z`; commit `6655c5d223a7852998cb6ec74e7e59b939feb221`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/6655c5d223a7852998cb6ec74e7e59b939feb221/policies/AWSGlueConsoleFullAccess
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/6655c5d223a7852998cb6ec74e7e59b939feb221/policies/AWSGlueConsoleFullAccess
  - Download SHA-256: `432c7034a949076bc054fbe71072c66e0cd7cbed98651462a62b2d99fb4ee1d3`.
- After: v13; AWS creation date `2023-03-30T15:58:41Z`; commit `08e36925dc6ef827eca182ec67061974c7aaa402`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/08e36925dc6ef827eca182ec67061974c7aaa402/policies/AWSGlueConsoleFullAccess
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/08e36925dc6ef827eca182ec67061974c7aaa402/policies/AWSGlueConsoleFullAccess
  - Download SHA-256: `7785143c9707b04a2b2bf14962e2d685af69de71d9a781ea9ee36713f1627265`.

None. Both PolicyVersion.Document objects are copied unchanged from the archive.

### PLH5-A03: AWSLambdaVPCAccessExecutionRole v2 -> v3

Expected: **pass**. Adds ec2:DescribeSubnets and a Sid. Existing network-interface writes are unchanged; the new action only describes subnet metadata.

- Before: v2; AWS creation date `2020-10-15T22:53:03Z`; commit `276d18501c8e97e341775d0b1953a49d56bd8f28`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/276d18501c8e97e341775d0b1953a49d56bd8f28/policies/AWSLambdaVPCAccessExecutionRole
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/276d18501c8e97e341775d0b1953a49d56bd8f28/policies/AWSLambdaVPCAccessExecutionRole
  - Download SHA-256: `624d4dbd5c24f3e1b7abb933daac350498c5150a637850d1bc99c0c4a87fc985`.
- After: v3; AWS creation date `2024-01-05T22:38:26Z`; commit `56bc2e692e1f8d4b62c6d089f55b39230cc437f3`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/56bc2e692e1f8d4b62c6d089f55b39230cc437f3/policies/AWSLambdaVPCAccessExecutionRole
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/56bc2e692e1f8d4b62c6d089f55b39230cc437f3/policies/AWSLambdaVPCAccessExecutionRole
  - Download SHA-256: `b72be9ab6cfe438c1d461d8268968d985240aad862a2541265f1f12453ebb45e`.

None. Both PolicyVersion.Document objects are copied unchanged from the archive.

### PLH5-A04: AWSLambda_ReadOnlyAccess v2 -> v3

Expected: **review**. Adds logs:StartLiveTail and logs:StopLiveTail only for /aws/lambda/* log groups. These logs were already readable; review the new live-stream access within that scope.

- Before: v2; AWS creation date `2023-07-27T17:32:05Z`; commit `9376f2efc867c81acc0e328c1052581a228dce0e`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/9376f2efc867c81acc0e328c1052581a228dce0e/policies/AWSLambda_ReadOnlyAccess
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/9376f2efc867c81acc0e328c1052581a228dce0e/policies/AWSLambda_ReadOnlyAccess
  - Download SHA-256: `2304fa1631d85185a4a40821f14fff5f34fd2d01d4dcddef3941a86d1f4acc71`.
- After: v3; AWS creation date `2025-03-17T21:07:06Z`; commit `2aba338d8f83d36cb6613929dac76de09d575ced`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/2aba338d8f83d36cb6613929dac76de09d575ced/policies/AWSLambda_ReadOnlyAccess
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/2aba338d8f83d36cb6613929dac76de09d575ced/policies/AWSLambda_ReadOnlyAccess
  - Download SHA-256: `a895eee11405c6217606f0e55c4319391591d48ec598d97c28e91e705a915a45`.

None. Both PolicyVersion.Document objects are copied unchanged from the archive.

### PLH5-A05: AmazonAthenaFullAccess v8 -> v9

Expected: **pass**. Adds pricing:GetProducts for the AWS price catalogue. This exposes public product/pricing metadata, not account data or write access.

- Before: v8; AWS creation date `2022-11-14T21:05:56Z`; commit `a4c74c9a2d1161aef841e20b7506ac9c293a353a`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/a4c74c9a2d1161aef841e20b7506ac9c293a353a/policies/AmazonAthenaFullAccess
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/a4c74c9a2d1161aef841e20b7506ac9c293a353a/policies/AmazonAthenaFullAccess
  - Download SHA-256: `9f5f2a12502216806227817879de8b66d7278ade7c6ba6001b2c024baf47c2dc`.
- After: v9; AWS creation date `2023-01-24T16:59:23Z`; commit `3de991771fa90e8f2f9dcfab9ed6fda27372cd42`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/3de991771fa90e8f2f9dcfab9ed6fda27372cd42/policies/AmazonAthenaFullAccess
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/3de991771fa90e8f2f9dcfab9ed6fda27372cd42/policies/AmazonAthenaFullAccess
  - Download SHA-256: `15dd0abfedca797dcb3be881f741183a7982892c2dbcd3789129401da9d77b5c`.

None. Both PolicyVersion.Document objects are copied unchanged from the archive.

### PLH5-A06: AmazonEC2ContainerRegistryReadOnly v2 -> v3

Expected: **pass**. Adds lifecycle-policy, lifecycle-preview, tag and scan-finding reads. Existing image pulls are unchanged and no repository write or new principal is added.

- Before: v2; AWS creation date `2016-10-11T22:08:43Z`; commit `ad4c2e8fdf77143515c3ca9f6efc0f4181ec6955`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/ad4c2e8fdf77143515c3ca9f6efc0f4181ec6955/policies/AmazonEC2ContainerRegistryReadOnly
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/ad4c2e8fdf77143515c3ca9f6efc0f4181ec6955/policies/AmazonEC2ContainerRegistryReadOnly
  - Download SHA-256: `6fa947a179295f447627d0a62768f73f82a84e376f96edc0d3a648e5ac380c7b`.
- After: v3; AWS creation date `2019-12-10T20:56:32Z`; commit `5d68ecfa21232031b8dea097737e4d3fb1808e4d`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/5d68ecfa21232031b8dea097737e4d3fb1808e4d/policies/AmazonEC2ContainerRegistryReadOnly
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/5d68ecfa21232031b8dea097737e4d3fb1808e4d/policies/AmazonEC2ContainerRegistryReadOnly
  - Download SHA-256: `1b86eef4a37d245b317474c630d36b2d6fbd4805c44bd5b23a79c0e5de93ee10`.

None. Both PolicyVersion.Document objects are copied unchanged from the archive.

### PLH5-A07: AmazonRDSFullAccess v10 -> v11

Expected: **pass**. Adds cloudwatch:ListMetrics only. Metric discovery does not add metric-value reads, database access or writes.

- Before: v10; AWS creation date `2022-12-19T20:09:11Z`; commit `86ef2d3c5544197673d4a9977ad79fd539b311f1`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/86ef2d3c5544197673d4a9977ad79fd539b311f1/policies/AmazonRDSFullAccess
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/86ef2d3c5544197673d4a9977ad79fd539b311f1/policies/AmazonRDSFullAccess
  - Download SHA-256: `a74f4a452aaeb0bc7a5b9ff276c8b28a994b86155ddb8c1f37eb9a87daba1834`.
- After: v11; AWS creation date `2023-03-15T13:44:54Z`; commit `45d10de95790add7451389df3a4481c257521879`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/45d10de95790add7451389df3a4481c257521879/policies/AmazonRDSFullAccess
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/45d10de95790add7451389df3a4481c257521879/policies/AmazonRDSFullAccess
  - Download SHA-256: `e638fcec4569d615a915848e0d8d2e432593cc69ece1e20027dc719d9df6b4db`.

None. Both PolicyVersion.Document objects are copied unchanged from the archive.

### PLH5-A08: AmazonS3ReadOnlyAccess v2 -> v3

Expected: **pass**. Adds s3:Describe* metadata actions. Existing s3:Get* and s3:List* data access is unchanged; no new principal or write permission is added.

- Before: v2; AWS creation date `2021-09-27T20:24:58Z`; commit `f1eeaa0507b249a144f0331d9c8d9fc84d20223b`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/f1eeaa0507b249a144f0331d9c8d9fc84d20223b/policies/AmazonS3ReadOnlyAccess
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/f1eeaa0507b249a144f0331d9c8d9fc84d20223b/policies/AmazonS3ReadOnlyAccess
  - Download SHA-256: `c454b834083661d625b47994750651b328540dc15061474d359aec38bd223968`.
- After: v3; AWS creation date `2023-08-10T21:31:39Z`; commit `cf1e4413c77fc0810114b72e35162510bce97756`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/cf1e4413c77fc0810114b72e35162510bce97756/policies/AmazonS3ReadOnlyAccess
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/cf1e4413c77fc0810114b72e35162510bce97756/policies/AmazonS3ReadOnlyAccess
  - Download SHA-256: `7269a302edcc46896fa987b10072f7044675db0fcef559a83dac68442658b559`.

None. Both PolicyVersion.Document objects are copied unchanged from the archive.

### PLH5-A09: AmazonSSMManagedInstanceCore v1 -> v2

Expected: **review**. Adds ssm:GetParameter on *. Parameter values are sensitive, but ssm:GetParameters already covered the same resources. Review the additional API without claiming a newly reachable secret set or a proven escalation.

- Before: v1; AWS creation date `2019-03-15T17:22:12Z`; commit `4bca9c8425f66d450210ba326afe300cfb98fb8e`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/4bca9c8425f66d450210ba326afe300cfb98fb8e/policies/AmazonSSMManagedInstanceCore
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/4bca9c8425f66d450210ba326afe300cfb98fb8e/policies/AmazonSSMManagedInstanceCore
  - Download SHA-256: `408301ad2af91843c2209d676aac74180613afb77d7a2152a77279806eda9bc6`.
- After: v2; AWS creation date `2019-05-23T16:54:21Z`; commit `caf80dacdf9b79240b96a614d40c85a9cd3c9c3b`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/caf80dacdf9b79240b96a614d40c85a9cd3c9c3b/policies/AmazonSSMManagedInstanceCore
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/caf80dacdf9b79240b96a614d40c85a9cd3c9c3b/policies/AmazonSSMManagedInstanceCore
  - Download SHA-256: `f0ecfbd1496d37120e6ffea4d0542e55850b1237746e79f3314451c2b26b506e`.

None. Both PolicyVersion.Document objects are copied unchanged from the archive.

### PLH5-A10: CloudWatchReadOnlyAccess v4 -> v5

Expected: **pass**. Adds oam:ListSinks and oam:ListAttachedLinks, with the latter limited to sink ARNs. These are observability configuration listings, not telemetry reads or writes.

- Before: v4; AWS creation date `2020-07-17T17:49:09Z`; commit `5811085d78f18a64460f9810b42671fea4141a05`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/5811085d78f18a64460f9810b42671fea4141a05/policies/CloudWatchReadOnlyAccess
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/5811085d78f18a64460f9810b42671fea4141a05/policies/CloudWatchReadOnlyAccess
  - Download SHA-256: `a98b3aa56ba7c76e427b4a184f0b6411e8114cae288872727ba919c057853dc1`.
- After: v5; AWS creation date `2022-11-27T13:08:46Z`; commit `33e468411837ccfc29feffb32b6ea8cb49b748f3`.
  - Snapshot: https://github.com/zoph-io/IAMTrail/blob/33e468411837ccfc29feffb32b6ea8cb49b748f3/policies/CloudWatchReadOnlyAccess
  - Raw: https://raw.githubusercontent.com/zoph-io/IAMTrail/33e468411837ccfc29feffb32b6ea8cb49b748f3/policies/CloudWatchReadOnlyAccess
  - Download SHA-256: `03f9ca687146cf6e12cf15679aa28af922fb5b040e802766321525cc94a4d81a`.

None. Both PolicyVersion.Document objects are copied unchanged from the archive.

## Modelled AWS and Rhino cases

### PLH5-A11: S3: remove the required KMS-key upload guardrail

Kind: **modelled**. Expected: **stop**.

Use the AWS specific-KMS-key Deny example unchanged for before; remove that statement for after.

Deleting the explicit deny permits otherwise-authorised uploads to use a different KMS key. This weakens the encryption-key requirement; it does not itself grant PutObject.

- https://docs.aws.amazon.com/AmazonS3/latest/userguide/example-bucket-policies.html

### PLH5-A12: KMS key policy: service-only grant creation becomes unrestricted

Kind: **modelled**. Expected: **stop**.

Wrap the AWS persistent-resource grant statement in a 2012-10-17 policy; remove its Condition only.

Removing kms:GrantIsForAWSResource permits the named key user to create grants directly. The key remains the policy attachment scope, but grant delegation is no longer restricted to integrated AWS services.

- https://docs.aws.amazon.com/kms/latest/developerguide/key-policy-default.html

### PLH5-A13: Secrets Manager: grant one named role read access to the attached secret

Kind: **modelled**. Expected: **review**.

Use the EC2RoleToAccessSecrets resource-policy example unchanged as after; before contains no statements.

A named role gains GetSecretValue on the secret carrying this resource policy. Resource * denotes that attached secret; confirm the role and business need rather than infer public access.

- https://docs.aws.amazon.com/secretsmanager/latest/userguide/auth-and-access_resource-policies.html

### PLH5-A14: SQS: SNS publisher loses the exact source-topic restriction

Kind: **modelled**. Expected: **stop**.

Wrap the AWS SNS-to-SQS example with Version 2012-10-17; remove aws:SourceArn for after.

The queue now accepts SendMessage from the SNS service without checking the source topic. The service principal remains fixed, but any eligible SNS topic can use the grant.

- https://docs.aws.amazon.com/sns/latest/dg/sns-access-policy-use-cases.html

### PLH5-A15: SNS identity policy: add Publish on one topic

Kind: **modelled**. Expected: **review**.

Convert the documented named-account Publish statement into an identity policy by removing Principal, retaining its action and topic ARN; compare against no grant.

Adds publishing to one named topic. This is scoped message injection authority requiring review; it does not change the topic policy or open access to everyone.

- https://docs.aws.amazon.com/sns/latest/dg/sns-access-policy-use-cases.html

### PLH5-A16: Lambda identity policy: invoke one function only through its authenticated URL

Kind: **modelled**. Expected: **review**.

Convert the documented authenticated function-URL resource policy to an identity policy by removing Principal from both statements; retain all ARNs, actions and conditions.

Both InvokeFunctionUrl and InvokeFunction are limited to one function. AWS_IAM and InvokedViaFunctionUrl conditions constrain the new invocation route; review this scoped capability.

- https://docs.aws.amazon.com/lambda/latest/dg/urls-auth.html

### PLH5-A17: ECR repository policy: existing named pull users gain image push

Kind: **modelled**. Expected: **review**.

After is the AWS AllowPushPull example unchanged. Before removes CompleteLayerUpload, InitiateLayerUpload, PutImage and UploadLayerPart, leaving the same users and pull checks.

The same two named users gain image upload and PutImage on the repository carrying this policy. Review the new deployment capability; Resource * is local to the attached repository, not all repositories.

- https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-policy-examples.html

### PLH5-A18: EC2: remove the IMDSv2 requirement from an instance launch policy

Kind: **modelled**. Expected: **stop**.

Combine the documented scoped RunInstances example and RequireImdsV2 deny for before; remove only the deny for after.

RunInstances remains authorised on the listed resources, but its explicit deny for missing required metadata tokens is removed. Launches using IMDSv1 become possible where service defaults permit them.

- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ExamplePolicies_EC2.html

### PLH5-A19: SSM Parameter Store: protected parameter overwrites become permitted

Kind: **modelled**. Expected: **stop**.

Use the AWS ssm:Overwrite example unchanged for before; delete its explicit deny for after.

The PutParameter allow already covers every parameter in the example account and region. Removing the ssm:Overwrite deny permits overwriting existing values, including sensitive configuration.

- https://docs.aws.amazon.com/systems-manager/latest/userguide/parameter-store-policy-conditions.html

### PLH5-A20: Rhino method 7: user-policy attachment loses its managed-policy allowlist

Kind: **modelled**. Expected: **stop**.

Use the AWS attach/detach managed-policy allowlist example unchanged for before; remove only Condition for after. Rhino supplies the escalation analysis.

AttachUserPolicy can now attach any managed policy to any user, including AdministratorAccess. This opens the policy-attachment escalation described by Rhino; the prior iam:PolicyARN allowlist blocked it.

- https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_iam_limit-managed.html
- https://rhinosecuritylabs.com/aws/aws-privilege-escalation-methods-mitigation/

### PLH5-A21: STS role trust: the named user can assume the role without MFA

Kind: **modelled**. Expected: **stop**.

Use the AWS CLI anika trust-policy example unchanged; remove the MFA Condition only.

The trusted user stays fixed, but removing aws:multifactorAuthPresent eliminates the MFA requirement for AssumeRole. No new principal is needed for this security regression.

- https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-role.html

### PLH5-A22: DynamoDB identity policy: add Query on Books and its indexes

Kind: **modelled**. Expected: **review**.

Project the documented Books table/index policy down to its existing dynamodb:Query action; retain both resource ARNs; compare against no grant.

Query can now return data from one named table and its indexes. This is new scoped data access; review its need without treating the index suffix as every table.

- https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/iam-policy-specific-table-indexes.html

### PLH5-A23: RDS IAM authentication: add one database-user login

Kind: **modelled**. Expected: **review**.

Use the single-DB-user AWS example unchanged as after; before has no statements.

Adds rds-db:connect for db_user on one DB resource ID. Successful login still requires database-side IAM authentication and user configuration; the IAM grant warrants review.

- https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.IAMDBAuth.IAMPolicy.html

### PLH5-A24: EventBridge event bus: add one named external account publisher

Kind: **modelled**. Expected: **stop**.

Use the AWS default-event-bus cross-account policy unchanged as after; before has no statements. Preserve the account ID in the JSON, even though surrounding AWS prose uses a different illustrative ID.

The event bus gains a cross-account PutEvents grant to the named account root principal. External event injection crosses the account boundary; a narrow bus ARN does not remove that risk.

- https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-event-bus-example-policy-cross-account.html

### PLH5-A25: Glue Data Catalog: add table creation in one named database

Kind: **modelled**. Expected: **review**.

Combine the AWS read-Books example and the AWS scoped CreateTable example for after; before contains the read example only. Both are published examples from the same page.

Adds CreateTable for db1/tbl1 with the required database and catalogue ancestors. This is scoped metadata creation, not DevEndpoint creation, PassRole or access to every database.

- https://docs.aws.amazon.com/glue/latest/dg/security_iam_id-based-policy-examples.html

### PLH5-A26: KMS key policy: remove both required encryption-context values

Kind: **modelled**. Expected: **stop**.

Wrap the AWS Department/Project Decrypt statement as a policy; remove Condition for after.

The named application role can decrypt without the Department=IT and Project=Alpha encryption-context checks. The policy no longer isolates this key use by that context.

- https://docs.aws.amazon.com/kms/latest/developerguide/conditions-kms.html

### PLH5-A27: Secrets Manager: remove the VPC-endpoint retrieval guardrail

Kind: **modelled**. Expected: **stop**.

Combine the AWS named-role GetSecretValue allow and sourceVpce deny examples; remove only the deny for after.

The named role retains GetSecretValue, but its attached secret no longer denies requests outside vpce-12345678. Otherwise-authorised off-endpoint reads are now possible.

- https://docs.aws.amazon.com/secretsmanager/latest/userguide/auth-and-access_resource-policies.html

### PLH5-A28: SQS: restrict an IP-based all-actions grant to send-only and block one address

Kind: **modelled**. Expected: **pass**.

Compare the two AWS published IP examples directly: before is the all-actions /24 grant; after is the send-only grant with an IP exclusion and explicit deny. No policy values are invented.

Within the same queue and /24 network, Allow narrows from sqs:* to SendMessage, excludes one address and adds an explicit deny for that address. No permission is broadened.

- https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-basic-examples-of-sqs-policies.html

### PLH5-A29: SNS resource policy: bind S3 publishing to its source account

Kind: **modelled**. Expected: **pass**.

After is the published S3-to-SNS source-account example with Version 2012-10-17. Before removes only its Condition.

Adding AWS:SourceAccount confines the existing S3 service grant to one source account. Topic, action and service principal remain unchanged.

- https://docs.aws.amazon.com/sns/latest/dg/sns-access-policy-use-cases.html

### PLH5-A30: Rhino method 16: add a DynamoDB-stream trigger route for new Lambda code

Kind: **modelled**. Expected: **stop**.

Reconstruct only the IAM action combination enumerated by Rhino method 16, using Resource * as the explicitly modelled unrestricted scope. Before has PassRole/CreateFunction; after adds CreateEventSourceMapping. This is not a captured deployed policy.

CreateEventSourceMapping completes a documented execution route for an actor already able to pass roles and create Lambda functions. Exploitation still requires a usable stream and suitable execution-role permissions; direct InvokeFunction is not required.

- https://rhinosecuritylabs.com/aws/aws-privilege-escalation-methods-mitigation/

## GMPP publications and data downloads

Every link in the ProjectLens source fields is listed here. All files in this section use the Open Government Licence v3.0. Case fields identify the exact project record/column and the file SHA-256.

### 2012-13: HO_Major_projects.csv

- Publication: https://www.gov.uk/government/publications/home-office-major-projects-portfolio-data-2013
- CSV: https://assets.publishing.service.gov.uk/media/5a74f33c40f0b6399b2af792/HO_Major_projects.csv
- SHA-256: `a6cd77f80e244d21121701703af1942a0c391871d03f5a51699ccebf556af5d8`.
- Cases: PLH5-B05, PLH5-B06.

### 2012-13: gmpp-data-MoJ-q2-2012-13.csv

- Publication: https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-moj-2013--2
- CSV: https://assets.publishing.service.gov.uk/media/5a7b88a5ed915d4147620fed/gmpp-data-MoJ-q2-2012-13.csv
- SHA-256: `99508c923a01dd8be3cd58df5e24bf7ba296eea54b22d2da25c44005b13aa4d7`.
- Cases: PLH5-B07.

### 2012-13: gmpp-data-DCMS-q2-2012-13.csv

- Publication: https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-dcms-2013
- CSV: https://assets.publishing.service.gov.uk/media/5a7bffe7ed915d01ba1ca7f4/gmpp-data-DCMS-q2-2012-13.csv
- SHA-256: `f2cc82bfb9f0ac3654c8da984793b77184a92e0d156eaf93bccb2c1efc4402cc`.
- Cases: PLH5-B01.

### 2012-13: gmpp-data-DH-q2-2012-13.csv

- Publication: https://www.gov.uk/government/publications/government-major-projects-portfolio-data-for-department-of-health-2013--2
- CSV: https://assets.publishing.service.gov.uk/media/5a7cab85ed915d6969f46841/gmpp-data-DH-q2-2012-13.csv
- SHA-256: `f631a2722cf4dbd3a8df742701cd21d8295297441e2f5af3ddfb458e9a359d33`.
- Cases: PLH5-B09, PLH5-B40.

### 2013-14: Government_Major_Projects_Portfolio_data__September_2013__CSV_.csv

- Publication: https://www.gov.uk/government/publications/major-projects-authority-annual-report-2014
- CSV: https://assets.publishing.service.gov.uk/media/5a7ea47a40f0b62305b823ee/Government_Major_Projects_Portfolio_data__September_2013__CSV_.csv
- SHA-256: `67cb811e8bf3f807ecee8cf177e36036bb6c0810a7c6f4077364cfebabb302a1`.
- Cases: PLH5-B01, PLH5-B05, PLH5-B06, PLH5-B07, PLH5-B08, PLH5-B09, PLH5-B10, PLH5-B38, PLH5-B39, PLH5-B40.

### 2014-15: GMPP_data_september_2014.csv

- Publication: https://www.gov.uk/government/publications/major-projects-authority-annual-report-2015
- CSV: https://assets.publishing.service.gov.uk/media/5a801db6ed915d74e33f88a0/GMPP_data_september_2014.csv
- SHA-256: `addde3f42fb6e4408c30e62d8cf3bf32eb5267d10802e13b3718faf051aef117`.
- Cases: PLH5-B08, PLH5-B10, PLH5-B38, PLH5-B39.

### 2015-16: government_major_projects_portfolio_data_september_2015.csv

- Publication: https://www.gov.uk/government/news/ipa-annual-report-on-major-government-projects-shows-improvements
- CSV: https://assets.publishing.service.gov.uk/media/5a80a49ce5274a2e8ab51547/government_major_projects_portfolio_data_september_2015.csv
- SHA-256: `c03928d2b93b02065869fb225a61ad88a44a1328c971a4c58e88349a74a32937`.
- Cases: PLH5-B03.

### 2016-17: Annual_Report_-_consolidated_data_and_narratives.csv

- Publication: https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2017
- CSV: https://assets.publishing.service.gov.uk/media/5a821ed6e5274a2e8ab57995/Annual_Report_-_consolidated_data_and_narratives.csv
- SHA-256: `1dc7c1598ab6955d5a1a704117b239350aef956b4b5aa141facaaadd331c6238`.
- Cases: PLH5-B03, PLH5-B04.

### 2017-18: csv-ipa-government-major-projects-portfolio-data-september-2017.csv

- Publication: https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2018
- CSV: https://assets.publishing.service.gov.uk/media/5b3e2e2340f0b678bc5d018c/csv-ipa-government-major-projects-portfolio-data-september-2017.csv
- SHA-256: `915cf52ff57a180c9f3989e9e29e56594efd81da1e66fa3757d66a145b71f726`.
- Cases: PLH5-B02, PLH5-B04, PLH5-B37.

### 2018-19: Annual_Report_on_Major_Projects_2018_to_2019__consolidated_data_and_narratives.csv

- Publication: https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2019
- CSV: https://assets.publishing.service.gov.uk/media/5d2dd08d40f0b64a8626b762/Annual_Report_on_Major_Projects_2018_to_2019__consolidated_data_and_narratives.csv
- SHA-256: `c22caa602ec3a7d9cc64cb20d559dbd5dbd0b9ceae5b4184675f247138d5b8b8`.
- Cases: PLH5-B02, PLH5-B27, PLH5-B37.

### 2019-20: GMPP_Government_Major_Projects_Portfolio_data_September_2019.csv

- Publication: https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2020
- CSV: https://assets.publishing.service.gov.uk/media/5f05de523a6f40041036a2ec/GMPP_Government_Major_Projects_Portfolio_data_September_2019.csv
- SHA-256: `0a9b082d6d1cd4c0887f94c6853b5e5ed5892c7c60a0d19ea9e7f839b9b7db89`.
- Cases: PLH5-B25, PLH5-B26, PLH5-B27, PLH5-B28.

### 2020-21: GMPP_Government_Major_Projects_Portofolio_AR_Data_March_2021.csv

- Publication: https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2021
- CSV: https://assets.publishing.service.gov.uk/media/60eecae48fa8f50c7ca55af1/GMPP_Government_Major_Projects_Portofolio_AR_Data_March_2021.csv
- SHA-256: `88a38d6ab1e1865a96aebde3d18e98150e79b4167b92b82dff15dcb96fa784f1`.
- Cases: PLH5-B23, PLH5-B24, PLH5-B25, PLH5-B26, PLH5-B28.

### 2021-22: GMPP_Government_Major_Projects_Portfolio_AR_Data_March_2022.csv

- Publication: https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2022
- CSV: https://assets.publishing.service.gov.uk/media/62d6c047e90e071e753d6936/GMPP_Government_Major_Projects_Portfolio_AR_Data_March_2022.csv
- SHA-256: `f34d32e4ed93a34cb9e7500491bba4cfa9568999c7dfabcbe74e0cc4cf538f2c`.
- Cases: PLH5-B22, PLH5-B23, PLH5-B24.

### 2022-23: GMPP_Government_Major_Projects_Portofolio_AR_Data_March_2023.csv

- Publication: https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2022-23
- CSV: https://assets.publishing.service.gov.uk/media/64b79c5171749c001389ee41/GMPP_Government_Major_Projects_Portofolio_AR_Data_March_2023.csv
- SHA-256: `ecd3756718a71e6f52fb25050caba535d64e7f5b21f6db361cfa10859a4364ac`.
- Cases: PLH5-B18, PLH5-B19, PLH5-B20, PLH5-B21, PLH5-B22, PLH5-B34.

### 2023-24: Government_Major_Projects_Portofolio_AR_Data_March_2024.csv

- Publication: https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2023-24
- CSV: https://assets.publishing.service.gov.uk/media/6787e8ee1124a2c3ceb646be/Government_Major_Projects_Portofolio_AR_Data_March_2024.csv
- SHA-256: `29596242964ff850351b9d091ae7db2584e1c8cdec3b7c8e53ca661479db614b`.
- Cases: PLH5-B11, PLH5-B12, PLH5-B13, PLH5-B14, PLH5-B15, PLH5-B16, PLH5-B17, PLH5-B18, PLH5-B19, PLH5-B20, PLH5-B21, PLH5-B29, PLH5-B30, PLH5-B31, PLH5-B32, PLH5-B33, PLH5-B34, PLH5-B35, PLH5-B36.

### 2024-25: nista_annual_report_data_2425.csv

- Publication: https://www.gov.uk/government/publications/nista-annual-report-2024-2025
- CSV: https://assets.publishing.service.gov.uk/media/6895fbb63080e72710b2e2ef/nista_annual_report_data_2425.csv
- SHA-256: `cac14ed4e6a90a81486bdc24295ea40996e7fd6fcd3834e3ffad8a2685f8c8b3`.
- Cases: PLH5-B11, PLH5-B12, PLH5-B13, PLH5-B14, PLH5-B15, PLH5-B16, PLH5-B17, PLH5-B29, PLH5-B30, PLH5-B31, PLH5-B32, PLH5-B33, PLH5-B35, PLH5-B36.

## Doubtful or scope-sensitive labels

These caveats are included in the ProjectLens case objects as `labelCaveat`. They must remain visible when reporting results; they are not undisclosed clean ground truth.

- **PLH5-B02 — Government Hubs Programme (contradiction)**: The narrative acknowledges individual and tranche movements. The label applies the C2 rule to its explicit programme-wide on-schedule claim and unstated programme-end extension.
- **PLH5-B05 — National Crime Agency (NCA) (contradiction)**: The successful launch and formal programme closure can be different milestones. This label follows the C2 rule for an unqualified delivered-to-schedule claim; it is not an allegation of late operational launch.
- **PLH5-B09 — E&N Herts NHS Trust - Lister Hospital ('Our Changing Hospitals' Phase 4 Programme) (contradiction)**: Consolidation is the named delivery outcome, rather than an explicit formal-closure date. The commentary and published end require interpretation of that scope; the label treats consolidation as the programme delivery finish.
- **PLH5-B10 — The Materiel Strategy (contradiction)**: The match-fit transformation end-state may precede administrative or supplier-contract closure. This label follows C2 by treating the stated transformation completion as the current programme finish; the scope distinction is uncertain.
- **PLH5-B40 — Mersey Care NHS LIFT scheme (TIME - To Improve Mental Health Environments) (published-date-check)**: The commentary explicitly dates the first part of a phased project, rather than an unqualified whole-project closure. It still places that completion after the published project end, so this is a date check rather than an assertion that the whole project closes in December 2015.

Policy calibration also needs care: A04 and A09 are deliberately review because the earlier policy already allowed the same logs/parameter resources through other APIs. A06 treats vulnerability findings and lifecycle configuration as metadata, which a stricter organisation could classify as sensitive. A30 requires a usable DynamoDB stream and adequate Lambda execution-role permissions; it demonstrates a newly enabled route, not proof that arbitrary role escalation will succeed in an unspecified account.

## Validation

All acceptance quotas, source hashes, source-cell equality, date arithmetic, exact managed-policy snapshots, duplicate checks and explicit name/ID exclusions passed. Neither target tool was executed.

Validation result retained at `.work/validation.json`. No accuracy score was calculated and no engine tuning was performed.
