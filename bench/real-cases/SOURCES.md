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
