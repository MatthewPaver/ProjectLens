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
