# Generated by Django 5.1.2 on 2024-10-16 12:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('redbox_core', '0052_chatllmbackend_display'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='redbox_tasks',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='user',
            name='business_unit',
            field=models.CharField(blank=True, choices=[('Borders Unit', 'Borders Unit'), ('Central Costs', 'Central Costs'), ('Central Digital and Data Office', 'Central Digital and Data Office'), ('Civil Service Commission', 'Civil Service Commission'), ('Civil Service Human Resources', 'Civil Service Human Resources'), ('CO Chief Operating Officer', 'CO Chief Operating Officer'), ('CO Digital', 'CO Digital'), ('CO HMT Commercial', 'CO HMT Commercial'), ('CO People and Places', 'CO People and Places'), ('CO Strategy, Finance, and Performance', 'CO Strategy Finance, and Performance'), ('Commercial Models', 'Commercial Models'), ('COP Presidency', 'COP Presidency'), ('Covid Inquiry', 'Covid Inquiry'), ('Crown Commercial Service', 'Crown Commercial Service'), ('CS Modernisation and Reform Unit', 'CS Modernisation and Reform Unit'), ('Delivery Group', 'Delivery Group'), ('Economic and Domestic Secretariat', 'Economic and Domestic Secretariat'), ('Equality and Human Rights Commission', 'Equality and Human Rights Commission'), ('Equality Hub', 'Equality Hub'), ('Flexible CS Pool', 'Flexible CS Pool'), ('Geospatial Commission', 'Geospatial Commission'), ('Government Business Services', 'Government Business Services'), ('Government Commercial and Grants Function', 'Government Commercial and Grants Function'), ('Government Communication Service', 'Government Communication Service'), ('Government Digital Service', 'Government Digital Service'), ('Government in Parliament', 'Government in Parliament'), ('Government Legal Department', 'Government Legal Department'), ('Government People Group', 'Government People Group'), ('Government Property Agency', 'Government Property Agency'), ('Government Security Group', 'Government Security Group'), ('Grenfell Inquiry', 'Grenfell Inquiry'), ('Infected Blood Inquiry', 'Infected Blood Inquiry'), ('Infrastructure and Projects Authority', 'Infrastructure and Projects Authority'), ('Inquiries Sponsorship Team', 'Inquiries Sponsorship Team'), ('Intelligence and Security Committee', 'Intelligence and Security Committee'), ('Joint Intelligence Organisation', 'Joint Intelligence Organisation'), ('National Security Secretariat', 'National Security Secretariat'), ("Office for Veterans' Affairs", "Office for Veterans' Affairs"), ('Office of Government Property', 'Office of Government Property'), ('Office of the Registrar of Consultant Lobbyists', 'Office of the Registrar of Consultant Lobbyists'), ("Prime Minister's Office", "Prime Minister's Office"), ('Propriety and Constitution Group', 'Propriety and Constitution Group'), ('Public Bodies and Priority Projects Unit', 'Public Bodies and Priority Projects Unit'), ('Public Inquiry Response Unit', 'Public Inquiry Response Unit'), ('Public Sector Fraud Authority', 'Public Sector Fraud Authority'), ('UKSV', 'UKSV'), ('Union and Constitution Group', 'Union and Constitution Group'), ('Other', 'Other')], max_length=64, null=True),
        ),
    ]