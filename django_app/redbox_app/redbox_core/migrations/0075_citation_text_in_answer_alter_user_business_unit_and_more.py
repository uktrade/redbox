# Generated by Django 5.1.3 on 2024-12-05 11:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("redbox_core", "0074_citation_text_in_answer_user_first_name_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="user",
            name="business_unit",
            field=models.CharField(
                blank=True,
                choices=[
                    (
                        "Competition, Markets and Regulatory Reform (CMRR)",
                        "Competition, Markets and Regulatory Reform (CMRR)",
                    ),
                    (
                        "Corporate Services Group (CSG)",
                        "Corporate Services Group (CSG)",
                    ),
                    (
                        "Trade Policy Implementation and Negotiations (TPIN)",
                        "Trade Policy Implementation and Negotiations (TPIN)",
                    ),
                    (
                        "Economic Security and Trade Relations (ESTR)",
                        "Economic Security and Trade Relations (ESTR)",
                    ),
                    ("Strategy and Investment", "Strategy and Investment"),
                    (
                        "Domestic and International Markets and Exports Group (DIME) UK Teams",
                        "Domestic and International Markets and Exports Group (DIME) UK Teams",
                    ),
                    ("Business Group", "Business Group"),
                    ("Overseas Regions", "Overseas Regions"),
                    ("Industrial Strategy Unit", "Industrial Strategy Unit"),
                    (
                        "Digital, Data and Technology (DDaT)",
                        "Digital, Data and Technology (DDaT)",
                    ),
                ],
                max_length=100,
                null=True,
            ),
        ),
        migrations.AlterField(
            model_name="user",
            name="profession",
            field=models.CharField(
                blank=True,
                choices=[
                    ("AN", "Analysis"),
                    ("CMC", "Commercial"),
                    ("COM", "Communications"),
                    ("CON", "Consular"),
                    ("CF", "Counter Fraud"),
                    ("DM", "Debt Management"),
                    ("DDT", "Digital, Data and Technology"),
                    ("FIN", "Finance"),
                    ("GM", "Grants Management"),
                    ("HR", "Human Resources"),
                    ("IA", "Intelligence Analysis"),
                    ("IAU", "Internal Audit"),
                    ("IT", "International Trade"),
                    ("KIM", "Knowledge and Information Management"),
                    ("LG", "Legal"),
                    ("OD", "Operational Delivery"),
                    ("POL", "Policy"),
                    ("PD", "Project Delivery"),
                    ("PROP", "Property"),
                    ("SC", "Security"),
                    ("SE", "Science and Engineering"),
                    ("OT", "Other"),
                ],
                max_length=4,
                null=True,
            ),
        ),
    ]